/* Sushil Dubey, Shashi Dugad, TIFR
*
*/


#ifndef RAWTODIGI_CPU_GPU_H
#define RAWTODIGI_CPU_GPU_H

// wrapper function to call RawToDigi on the GPU from host side
void RawToDigi_wrapper (const uint wordCounter, uint *word, const uint fedCounter,  uint *fedIndex,
                        uint *eventIndex, bool convertADCtoElectrons, uint *xx_h, uint *yy_h, uint *adc_h, int *mIndexStart_h,
                        int *mIndexEnd_h);

void initCablingMap();
void initDeviceMemory();
void freeMemory();

// reference cmssw/RecoLocalTracker/SiPixelClusterizer
// all are runtime const, should be specified in python _cfg.py
struct ADCThreshold {
  const int thePixelThreshold = 1000; // default Pixel threshold in electrons
  const int theSeedThreshold = 1000; //seed thershold in electrons not used in our algo
  const float theClusterThreshold = 4000; // Cluster threshold in electron
  const int ConversionFactor = 65;  // adc to electron conversion factor
  
  // following are the default value
  // it should be i python script
  const int theStackADC_  = 255; // the maximum adc count for stack layer
  const int theFirstStack_ = 5; // the index of the fits stack layer
  const double theElectronPerADCGain_ = 600; //ADC to electron conversion

};
#endif
