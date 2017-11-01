/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

// This sample needs at least CUDA 5.5 and a GPU that has at least Compute Capability 2.0

// This sample demonstrates a simple image processing pipeline.
// First, a JPEG file is huffman decoded and inverse DCT transformed and dequantized.
// Then the different planes are resized. Finally, the resized image is quantized, forward
// DCT transformed and huffman encoded.

#include <npp.h>
#include <cuda_runtime.h>
#include <Exceptions.h>

#include "Endianess.h"
#include <math.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <helper_string.h>
#include <helper_cuda.h>

using namespace std;

struct FrameHeader
{
    unsigned char nSamplePrecision;
    unsigned short nHeight;
    unsigned short nWidth;
    unsigned char nComponents;
    unsigned char aComponentIdentifier[3];
    unsigned char aSamplingFactors[3];
    unsigned char aQuantizationTableSelector[3];
};

struct ScanHeader
{
    unsigned char nComponents;
    unsigned char aComponentSelector[3];
    unsigned char aHuffmanTablesSelector[3];
    unsigned char nSs;
    unsigned char nSe;
    unsigned char nA;
};

struct QuantizationTable
{
    unsigned char nPrecisionAndIdentifier;
    unsigned char aTable[64];
};

struct HuffmanTable
{
    unsigned char nClassAndIdentifier;
    unsigned char aCodes[16];
    unsigned char aTable[256];
};


int DivUp(int x, int d)
{
    return (x + d - 1) / d;
}

template<typename T>
T readAndAdvance(const unsigned char *&pData)
{
    T nElement = readBigEndian<T>(pData);
    pData += sizeof(T);
    return nElement;
}

template<typename T>
void writeAndAdvance(unsigned char *&pData, T nElement)
{
    writeBigEndian<T>(pData, nElement);
    pData += sizeof(T);
}


int nextMarker(const unsigned char *pData, int &nPos, int nLength)
{
    unsigned char c = pData[nPos++];

    do
    {
        while (c != 0xffu && nPos < nLength)
        {
            c =  pData[nPos++];
        }

        if (nPos >= nLength)
            return -1;

        c =  pData[nPos++];
    }
    while (c == 0 || c == 0x0ffu);

    return c;
}

void writeMarker(unsigned char nMarker, unsigned char *&pData)
{
    *pData++ = 0x0ff;
    *pData++ = nMarker;
}

void writeJFIFTag(unsigned char *&pData)
{
    const char JFIF_TAG[] =
    {
        0x4a, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x02,
        0x00,
        0x00, 0x01, 0x00, 0x01,
        0x00, 0x00
    };

    writeMarker(0x0e0, pData);
    writeAndAdvance<unsigned short>(pData, sizeof(JFIF_TAG) + sizeof(unsigned short));
    memcpy(pData, JFIF_TAG, sizeof(JFIF_TAG));
    pData += sizeof(JFIF_TAG);
}

void loadJpeg(const char *input_file, unsigned char *&pJpegData, int &nInputLength)
{
    // Load file into CPU memory
    ifstream stream(input_file, ifstream::binary);

    if (!stream.good())
    {
        return;
    }

    stream.seekg(0, ios::end);
    nInputLength = (int)stream.tellg();
    stream.seekg(0, ios::beg);

    pJpegData = new unsigned char[nInputLength];
    stream.read(reinterpret_cast<char *>(pJpegData), nInputLength);
}

void readFrameHeader(const unsigned char *pData, FrameHeader &header)
{
    readAndAdvance<unsigned short>(pData);
    header.nSamplePrecision = readAndAdvance<unsigned char>(pData);
    header.nHeight = readAndAdvance<unsigned short>(pData);
    header.nWidth = readAndAdvance<unsigned short>(pData);
    header.nComponents = readAndAdvance<unsigned char>(pData);

    for (int c=0; c<header.nComponents; ++c)
    {
        header.aComponentIdentifier[c] = readAndAdvance<unsigned char>(pData);
        header.aSamplingFactors[c] = readAndAdvance<unsigned char>(pData);
        header.aQuantizationTableSelector[c] = readAndAdvance<unsigned char>(pData);
    }

}

void writeFrameHeader(const FrameHeader &header, unsigned char *&pData)
{
    unsigned char aTemp[128];
    unsigned char *pTemp = aTemp;

    writeAndAdvance<unsigned char>(pTemp, header.nSamplePrecision);
    writeAndAdvance<unsigned short>(pTemp, header.nHeight);
    writeAndAdvance<unsigned short>(pTemp, header.nWidth);
    writeAndAdvance<unsigned char>(pTemp, header.nComponents);

    for (int c=0; c<header.nComponents; ++c)
    {
        writeAndAdvance<unsigned char>(pTemp,header.aComponentIdentifier[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aSamplingFactors[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aQuantizationTableSelector[c]);
    }

    unsigned short nLength = (unsigned short)(pTemp - aTemp);

    writeMarker(0x0C0, pData);
    writeAndAdvance<unsigned short>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}


void readScanHeader(const unsigned char *pData, ScanHeader &header)
{
    readAndAdvance<unsigned short>(pData);

    header.nComponents = readAndAdvance<unsigned char>(pData);

    for (int c=0; c<header.nComponents; ++c)
    {
        header.aComponentSelector[c] = readAndAdvance<unsigned char>(pData);
        header.aHuffmanTablesSelector[c] = readAndAdvance<unsigned char>(pData);
    }

    header.nSs = readAndAdvance<unsigned char>(pData);
    header.nSe = readAndAdvance<unsigned char>(pData);
    header.nA = readAndAdvance<unsigned char>(pData);
}


void writeScanHeader(const ScanHeader &header, unsigned char *&pData)
{
    unsigned char aTemp[128];
    unsigned char *pTemp = aTemp;

    writeAndAdvance<unsigned char>(pTemp, header.nComponents);

    for (int c=0; c<header.nComponents; ++c)
    {
        writeAndAdvance<unsigned char>(pTemp,header.aComponentSelector[c]);
        writeAndAdvance<unsigned char>(pTemp,header.aHuffmanTablesSelector[c]);
    }

    writeAndAdvance<unsigned char>(pTemp,  header.nSs);
    writeAndAdvance<unsigned char>(pTemp,  header.nSe);
    writeAndAdvance<unsigned char>(pTemp,  header.nA);

    unsigned short nLength = (unsigned short)(pTemp - aTemp);

    writeMarker(0x0DA, pData);
    writeAndAdvance<unsigned short>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}


void readQuantizationTables(const unsigned char *pData, QuantizationTable *pTables)
{
    unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

    while (nLength > 0)
    {
        unsigned char nPrecisionAndIdentifier = readAndAdvance<unsigned char>(pData);
        int nIdentifier = nPrecisionAndIdentifier & 0x0f;

        pTables[nIdentifier].nPrecisionAndIdentifier = nPrecisionAndIdentifier;
        memcpy(pTables[nIdentifier].aTable, pData, 64);
        pData += 64;

        nLength -= 65;
    }
}

void writeQuantizationTable(const QuantizationTable &table, unsigned char *&pData)
{
    writeMarker(0x0DB, pData);
    writeAndAdvance<unsigned short>(pData, sizeof(QuantizationTable) + 2);
    memcpy(pData, &table, sizeof(QuantizationTable));
    pData += sizeof(QuantizationTable);
}

void readHuffmanTables(const unsigned char *pData, HuffmanTable *pTables)
{
    unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

    while (nLength > 0)
    {
        unsigned char nClassAndIdentifier = readAndAdvance<unsigned char>(pData);
        int nClass = nClassAndIdentifier >> 4; // AC or DC
        int nIdentifier = nClassAndIdentifier & 0x0f;
        int nIdx = nClass * 2 + nIdentifier;
        pTables[nIdx].nClassAndIdentifier = nClassAndIdentifier;

        // Number of Codes for Bit Lengths [1..16]
        int nCodeCount = 0;

        for (int i = 0; i < 16; ++i)
        {
            pTables[nIdx].aCodes[i] = readAndAdvance<unsigned char>(pData);
            nCodeCount += pTables[nIdx].aCodes[i];
        }

        memcpy(pTables[nIdx].aTable, pData, nCodeCount);
        pData += nCodeCount;

        nLength -= (17 + nCodeCount);
    }
}

void writeHuffmanTable(const HuffmanTable &table, unsigned char *&pData)
{
    writeMarker(0x0C4, pData);

    // Number of Codes for Bit Lengths [1..16]
    int nCodeCount = 0;

    for (int i = 0; i < 16; ++i)
    {
        nCodeCount += table.aCodes[i];
    }

    writeAndAdvance<unsigned short>(pData, 17 + nCodeCount + 2);
    memcpy(pData, &table, 17 + nCodeCount);
    pData += 17 + nCodeCount;
}


void readRestartInterval(const unsigned char *pData, int &nRestartInterval)
{
    readAndAdvance<unsigned short>(pData);
    nRestartInterval = readAndAdvance<unsigned short>(pData);
}

void printHelp()
{
    cout << "jpegNPP usage" << endl;
    cout << "   -input=srcfile.jpg     (input  file JPEG image)" << endl;
    cout << "   -output=destfile.jpg   (output file JPEG image)" << endl;
    cout << "   -scale=1.0             (scale multiplier for width and height)" << endl << endl;
}

bool printfNPPinfo(int argc, char *argv[], int cudaVerMajor, int cudaVerMinor)
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    bool bVal = checkCudaCapabilities(cudaVerMajor, cudaVerMinor);
    return bVal;
}

int main(int argc, char **argv)
{
    // Min spec is SM 2.0 devices
    if (printfNPPinfo(argc, argv, 2, 0) == false)
    {
        cerr << "jpegNPP requires a GPU with Compute Capability 2.0 or higher" << endl;
        return EXIT_SUCCESS;
    }

    const char *szInputFile;
    const char *szOutputFile;
    float nScaleFactor;

    if ((argc == 1) || checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printHelp();
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input", (char **)&szInputFile);
    }
    else
    {
        szInputFile = sdkFindFilePath("Growth_of_cubic_bacteria_25x16.jpg", argv[0]);
    }

    cout << "Source File: " << szInputFile << endl;

    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "output", (char **)&szOutputFile);
    }
    else
    {
        szOutputFile = "scaled.jpg";
    }

    cout << "Output File: " << szOutputFile << endl;

    if (checkCmdLineFlag(argc, (const char **)argv, "scale"))
    {
        nScaleFactor = max(0.0f, min(getCmdLineArgumentFloat(argc, (const char **)argv, "scale"), 1.0f));
    }
    else
    {
        nScaleFactor = 0.5f;
    }

    cout << "Scale Factor: " << nScaleFactor << endl;

    NppiDCTState *pDCTState;
    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));

    unsigned char *pJpegData = 0;
    int nInputLength = 0;

    // Load Jpeg
    loadJpeg(szInputFile, pJpegData, nInputLength);

    if (pJpegData == 0)
    {
        cerr << "Input File Error: " << szInputFile << endl;
        return EXIT_FAILURE;
    }

    /***************************
    *
    *   Input
    *
    ***************************/


    // Check if this is a valid JPEG file
    int nPos = 0;
    int nMarker = nextMarker(pJpegData, nPos, nInputLength);

    if (nMarker != 0x0D8)
    {
        cerr << "Invalid Jpeg Image" << endl;
        return EXIT_FAILURE;
    }

    nMarker = nextMarker(pJpegData, nPos, nInputLength);

    // Parsing and Huffman Decoding (on host)
    FrameHeader oFrameHeader;
    QuantizationTable aQuantizationTables[4];
    Npp8u *pdQuantizationTables;
    cudaMalloc(&pdQuantizationTables, 64 * 4);

    HuffmanTable aHuffmanTables[4];
    HuffmanTable *pHuffmanDCTables = aHuffmanTables;
    HuffmanTable *pHuffmanACTables = &aHuffmanTables[2];
    ScanHeader oScanHeader;
    memset(&oFrameHeader,0,sizeof(FrameHeader));
    memset(aQuantizationTables,0, 4 * sizeof(QuantizationTable));
    memset(aHuffmanTables,0, 4 * sizeof(HuffmanTable));
    int nMCUBlocksH = 0;
    int nMCUBlocksV = 0;

    int nRestartInterval = -1;

    NppiSize aSrcSize[3];
    Npp16s *aphDCT[3] = {0,0,0};
    Npp16s *apdDCT[3] = {0,0,0};
    Npp32s aDCTStep[3];

    Npp8u *apSrcImage[3] = {0,0,0};
    Npp32s aSrcImageStep[3];

    Npp8u *apDstImage[3] = {0,0,0};
    Npp32s aDstImageStep[3];
    NppiSize aDstSize[3];

    while (nMarker != -1)
    {
        if (nMarker == 0x0D8)
        {
            // Embedded Thumbnail, skip it
            int nNextMarker = nextMarker(pJpegData, nPos, nInputLength);

            while (nNextMarker != -1 && nNextMarker != 0x0D9)
            {
                nNextMarker = nextMarker(pJpegData, nPos, nInputLength);
            }
        }

        if (nMarker == 0x0DD)
        {
            readRestartInterval(pJpegData + nPos, nRestartInterval);
        }

        if ((nMarker == 0x0C0) | (nMarker == 0x0C2))
        {
            //Assert Baseline for this Sample
            //Note: NPP does support progressive jpegs for both encode and decode
            if (nMarker != 0x0C0)
            {
                cerr << "The sample does only support baseline JPEG images" << endl;
                return EXIT_SUCCESS;
            }

            // Baseline or Progressive Frame Header
            readFrameHeader(pJpegData + nPos, oFrameHeader);
            cout << "Image Size: " << oFrameHeader.nWidth << "x" << oFrameHeader.nHeight << "x" << static_cast<int>(oFrameHeader.nComponents) << endl;

            //Assert 3-Channel Image for this Sample
            if (oFrameHeader.nComponents != 3)
            {
                cerr << "The sample does only support color JPEG images" << endl;
                return EXIT_SUCCESS;
            }

            // Compute channel sizes as stored in the JPEG (8x8 blocks & MCU block layout)
            for (int i=0; i < oFrameHeader.nComponents; ++i)
            {
                nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f );
                nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4 );
            }

            for (int i=0; i < oFrameHeader.nComponents; ++i)
            {
                NppiSize oBlocks;
                NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i]  >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};

                oBlocks.width = (int)ceil((oFrameHeader.nWidth + 7)/8  *
                                          static_cast<float>(oBlocksPerMCU.width)/nMCUBlocksH);
                oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

                oBlocks.height = (int)ceil((oFrameHeader.nHeight+7)/8 *
                                           static_cast<float>(oBlocksPerMCU.height)/nMCUBlocksV);
                oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

                aSrcSize[i].width = oBlocks.width * 8;
                aSrcSize[i].height = oBlocks.height * 8;

                // Allocate Memory
                size_t nPitch;
                NPP_CHECK_CUDA(cudaMallocPitch(&apdDCT[i], &nPitch, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height));
                aDCTStep[i] = static_cast<Npp32s>(nPitch);

                NPP_CHECK_CUDA(cudaMallocPitch(&apSrcImage[i], &nPitch, aSrcSize[i].width, aSrcSize[i].height));
                aSrcImageStep[i] = static_cast<Npp32s>(nPitch);

                NPP_CHECK_CUDA(cudaHostAlloc(&aphDCT[i], aDCTStep[i] * oBlocks.height, cudaHostAllocDefault));
            }
        }

        if (nMarker == 0x0DB)
        {
            // Quantization Tables
            readQuantizationTables(pJpegData + nPos, aQuantizationTables);
        }

        if (nMarker == 0x0C4)
        {
            // Huffman Tables
            readHuffmanTables(pJpegData + nPos, aHuffmanTables);
        }

        if (nMarker == 0x0DA)
        {
            // Scan
            readScanHeader(pJpegData + nPos, oScanHeader);
            nPos += 6 + oScanHeader.nComponents * 2;

            int nAfterNextMarkerPos = nPos;
            int nAfterScanMarker = nextMarker(pJpegData, nAfterNextMarkerPos, nInputLength);

            if (nRestartInterval > 0)
            {
                while (nAfterScanMarker >= 0x0D0 && nAfterScanMarker <= 0x0D7)
                {
                    // This is a restart marker, go on
                    nAfterScanMarker = nextMarker(pJpegData, nAfterNextMarkerPos, nInputLength);
                }
            }

            NppiDecodeHuffmanSpec *apHuffmanDCTable[3];
            NppiDecodeHuffmanSpec *apHuffmanACTable[3];

            for (int i = 0; i < 3; ++i)
            {
                nppiDecodeHuffmanSpecInitAllocHost_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &apHuffmanDCTable[i]);
                nppiDecodeHuffmanSpecInitAllocHost_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &apHuffmanACTable[i]);
            }

            NPP_CHECK_NPP(nppiDecodeHuffmanScanHost_JPEG_8u16s_P3R(pJpegData + nPos, nAfterNextMarkerPos - nPos - 2,
                                                                   nRestartInterval, oScanHeader.nSs, oScanHeader.nSe, 
                                                                   oScanHeader.nA >> 4, oScanHeader.nA & 0x0f,
                                                                   aphDCT,  aDCTStep,
                                                                   apHuffmanDCTable,
                                                                   apHuffmanACTable,
                                                                   aSrcSize));

            for (int i = 0; i < 3; ++i)
            {
                nppiDecodeHuffmanSpecFreeHost_JPEG(apHuffmanDCTable[i]);
                nppiDecodeHuffmanSpecFreeHost_JPEG(apHuffmanACTable[i]);
            }
        }

        nMarker = nextMarker(pJpegData, nPos, nInputLength);
    }

    // Copy DCT coefficients and Quantization Tables from host to device 
    Npp8u aZigzag[] = {
            0,  1,  5,  6, 14, 15, 27, 28,
            2,  4,  7, 13, 16, 26, 29, 42,
            3,  8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
    };

    for (int i = 0; i < 4; ++i)
    {
        Npp8u temp[64];

        for( int k = 0 ; k < 32 ; ++k )
        {
            temp[2 * k + 0] = aQuantizationTables[i].aTable[aZigzag[k +  0]];
            temp[2 * k + 1] = aQuantizationTables[i].aTable[aZigzag[k + 32]];
        }

        NPP_CHECK_CUDA(cudaMemcpyAsync((unsigned char *)pdQuantizationTables + i * 64, temp, 64, cudaMemcpyHostToDevice));

           
    }
        

    for (int i = 0; i < 3; ++i)
    {
        NPP_CHECK_CUDA(cudaMemcpyAsync(apdDCT[i], aphDCT[i], aDCTStep[i] * aSrcSize[i].height / 8, cudaMemcpyHostToDevice));
    }

    // Inverse DCT
    for (int i = 0; i < 3; ++i)
    {
        NPP_CHECK_NPP(nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW(apdDCT[i], aDCTStep[i],
                                                              apSrcImage[i], aSrcImageStep[i],
                                                              pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
                                                              aSrcSize[i],
                                                              pDCTState));
    }


    /***************************
    *
    *   Processing
    *
    ***************************/

    // Compute channel sizes as stored in the output JPEG (8x8 blocks & MCU block layout)
    NppiSize oDstImageSize;
    float frameWidth = floor((float)oFrameHeader.nWidth * (float)nScaleFactor);
    float frameHeight = floor((float)oFrameHeader.nHeight * (float)nScaleFactor);

    oDstImageSize.width  = (int)max(1.0f, frameWidth);
    oDstImageSize.height = (int)max(1.0f, frameHeight);

    cout << "Output Size: " << oDstImageSize.width << "x" << oDstImageSize.height << "x" << static_cast<int>(oFrameHeader.nComponents) << endl;

    for (int i=0; i < oFrameHeader.nComponents; ++i)
    {
        NppiSize oBlocks;
        NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] & 0x0f, oFrameHeader.aSamplingFactors[i] >> 4};

        oBlocks.width = (int)ceil((oDstImageSize.width + 7)/8  *
                                  static_cast<float>(oBlocksPerMCU.width)/nMCUBlocksH);
        oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

        oBlocks.height = (int)ceil((oDstImageSize.height+7)/8 *
                                   static_cast<float>(oBlocksPerMCU.height)/nMCUBlocksV);
        oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

        aDstSize[i].width = oBlocks.width * 8;
        aDstSize[i].height = oBlocks.height * 8;

        // Allocate Memory
        size_t nPitch;
        NPP_CHECK_CUDA(cudaMallocPitch(&apDstImage[i], &nPitch, aDstSize[i].width, aDstSize[i].height));
        aDstImageStep[i] = static_cast<Npp32s>(nPitch);
    }

    // Scale to target image size
    // Assume we only deal with 420 images.
    int aSampleFactor[3] = {1, 2, 2};
    for (int i = 0; i < 3; ++i)
    {
        NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};
        NppiSize oSrcImageSize = {(oFrameHeader.nWidth * oBlocksPerMCU.width) / nMCUBlocksH, (oFrameHeader.nHeight * oBlocksPerMCU.height)/nMCUBlocksV};
        NppiRect oSrcImageROI = {0,0,oSrcImageSize.width, oSrcImageSize.height};
        NppiRect oDstImageROI;
        oDstImageROI.x = 0;
        oDstImageROI.y = 0;
        oDstImageROI.width = oDstImageSize.width / aSampleFactor[i];
        oDstImageROI.height = oDstImageSize.height / aSampleFactor[i];
        
        NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER;
        
        if (nScaleFactor >= 1.f)
            eInterploationMode = NPPI_INTER_LANCZOS;
        
        NPP_CHECK_NPP(nppiResize_8u_C1R(apSrcImage[i], aSrcImageStep[i], oSrcImageSize, oSrcImageROI,
                                        apDstImage[i], aDstImageStep[i], oDstImageSize, oDstImageROI, eInterploationMode));
    }

    /***************************
    *
    *   Output
    *
    ***************************/


    // Forward DCT
    for (int i = 0; i < 3; ++i)
    {
        NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apDstImage[i], aDstImageStep[i],
                                                              apdDCT[i], aDCTStep[i],
                                                              pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
                                                              aDstSize[i],
                                                              pDCTState));
    }


    // Huffman Encoding
    Npp8u *pdScan;
    Npp32s nScanSize;
    nScanSize = oDstImageSize.width * oDstImageSize.height * 2;
    nScanSize = nScanSize > (4 << 20) ? nScanSize : (4 << 20);    
    NPP_CHECK_CUDA(cudaMalloc(&pdScan, nScanSize));

    Npp8u *pJpegEncoderTemp;
    size_t nTempSize;
    NPP_CHECK_NPP(nppiEncodeHuffmanGetSize(aSrcSize[0], 3, &nTempSize));
    NPP_CHECK_CUDA(cudaMalloc(&pJpegEncoderTemp, nTempSize));

    NppiEncodeHuffmanSpec *apHuffmanDCTable[3];
    NppiEncodeHuffmanSpec *apHuffmanACTable[3];

    for (int i = 0; i < 3; ++i)
    {
        nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &apHuffmanDCTable[i]);
        nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &apHuffmanACTable[i]);
    }
    
    Npp8u * hpCodesDC[3];
    Npp8u * hpCodesAC[3];
    Npp8u * hpTableDC[3];
    Npp8u * hpTableAC[3];
    for(int iComponent = 0; iComponent < 2; ++ iComponent)
    {
        hpCodesDC[iComponent] = pHuffmanDCTables[iComponent].aCodes;
        hpCodesAC[iComponent] = pHuffmanACTables[iComponent].aCodes;
        hpTableDC[iComponent] = pHuffmanDCTables[iComponent].aTable;
        hpTableAC[iComponent] = pHuffmanACTables[iComponent].aTable;
    }
    
    Npp32s nScanLength;                
    NPP_CHECK_NPP(nppiEncodeOptimizeHuffmanScan_JPEG_8u16s_P3R(apdDCT, aDCTStep,
                                                       0, oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA >> 4, oScanHeader.nA & 0x0f,
                                                       pdScan, &nScanLength,
                                                       hpCodesDC, hpTableDC, hpCodesAC, hpTableAC,
                                                       apHuffmanDCTable,
                                                       apHuffmanACTable,
                                                       aDstSize,
                                                       pJpegEncoderTemp));

    for (int i = 0; i < 3; ++i)
    {
        nppiEncodeHuffmanSpecFree_JPEG(apHuffmanDCTable[i]);
        nppiEncodeHuffmanSpecFree_JPEG(apHuffmanACTable[i]);
    }

    // Write JPEG
    unsigned char *pDstJpeg = new unsigned char[nScanSize];
    unsigned char *pDstOutput = pDstJpeg;

    oFrameHeader.nWidth = oDstImageSize.width;
    oFrameHeader.nHeight = oDstImageSize.height;

    writeMarker(0x0D8, pDstOutput);
    writeJFIFTag(pDstOutput);
    writeQuantizationTable(aQuantizationTables[0], pDstOutput);
    writeQuantizationTable(aQuantizationTables[1], pDstOutput);
    writeFrameHeader(oFrameHeader, pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[1], pDstOutput);
    writeScanHeader(oScanHeader, pDstOutput);
    NPP_CHECK_CUDA(cudaMemcpy(pDstOutput, pdScan, nScanLength, cudaMemcpyDeviceToHost));
    pDstOutput += nScanLength;
    writeMarker(0x0D9, pDstOutput);

    {
        // Write result to file.
        std::ofstream outputFile(szOutputFile, ios::out | ios::binary);
        outputFile.write(reinterpret_cast<const char *>(pDstJpeg), static_cast<int>(pDstOutput - pDstJpeg));
    }

    // Cleanup
    delete [] pJpegData;
    delete [] pDstJpeg;

    cudaFree(pJpegEncoderTemp);
    cudaFree(pdQuantizationTables);
    cudaFree(pdScan);

    nppiDCTFree(pDCTState);

    for (int i = 0; i < 3; ++i)
    {
        cudaFree(apdDCT[i]);
        cudaFreeHost(aphDCT[i]);
        cudaFree(apSrcImage[i]);
        cudaFree(apDstImage[i]);
    }

    return EXIT_SUCCESS;
}
