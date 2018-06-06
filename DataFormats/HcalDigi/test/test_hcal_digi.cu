#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <assert.h>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/Common/interface/DataFrame.h"

__global__ void kernel_test_hcal_qiesample(HcalQIESample* sample, uint16_t value) {
    printf("kernel: testing hcal qie sampel\n");
    printf("%f %f %f\n", nominal_adc2fc[0], nominal_adc2fc[1], nominal_adc2fc[2]);

    HcalQIESample tmp{value};
    *sample = tmp;
}

__global__ void kernel_test_hcal_qie8_hbhedf(HBHEDataFrame *df) {
    printf("kernel: testing hcal hbhe dataframe\n");
    df->setSize(10);
    for (auto i=0; i<10; i++)
        df->setSample(i, HcalQIESample(100));
    df->setReadoutIds(HcalElectronicsId(100));
}

void test_hcal_qiesample() {
    HcalQIESample h_sample, h_test_sample0{100}, h_test_sample1;
    HcalQIESample *d_sample;

    cudaMalloc((void**)&d_sample, sizeof(HcalQIESample));
    cudaMemcpy(d_sample, &h_sample, sizeof(HcalQIESample), cudaMemcpyHostToDevice);
    kernel_test_hcal_qiesample<<<1,1>>>(d_sample, 100);
    cudaMemcpy(&h_sample, d_sample, sizeof(HcalQIESample), cudaMemcpyDeviceToHost);

    assert(h_sample() == h_test_sample0());
    assert(h_sample() != h_test_sample1());
}

template<typename TDF>
__global__ void kernel_test_hcal_qie8_digis(TDF *pdfs, uint32_t* out) {
    int id = threadIdx.x;
    uint32_t sum = 0;
    for (auto i=0; i<10; i++)
        sum += pdfs[id].sample(i).raw();
    out[id] = sum;
}

template<typename TDF>
__global__ void kernel_test_hcal_qie1011_digis(uint16_t* pdfs, uint32_t* out, int samples) {
    printf("kernel: testing hcal qie1011 df\n");
    int id = threadIdx.x;
    uint32_t sum=0;
    int nwords = TDF::WORDS_PER_SAMPLE*samples + TDF::HEADER_WORDS + TDF::FLAG_WORDS;
    TDF df(edm::DataFrame(0, pdfs + id*nwords, nwords));
    for (auto i=0; i< df.samples(); i++) {
        sum += df[i].adc();
    }

    out[id] = sum;
}

template<typename TDF>
void test_hcal_qie1011_digis() {
    constexpr int size = 10;
    constexpr int samples = 10;
    constexpr int detid = 2;
    HcalDataFrameContainer<TDF> coll{samples, detid};
    TDF *d_dfs;
    uint16_t *d_data;
    uint32_t *d_out;
    uint32_t h_out[size], h_test_out[size];
    for (auto i=0; i<size; i++) {
        // #words per single TDF
        uint16_t tmp[TDF::WORDS_PER_SAMPLE * samples + TDF::HEADER_WORDS + TDF::FLAG_WORDS];
        h_test_out[i] = 0;
        for (auto j=TDF::HEADER_WORDS; j<TDF::WORDS_PER_SAMPLE*samples + TDF::HEADER_WORDS; j++) {
            tmp[j] = 100;
        }
        TDF df(edm::DataFrame(0, tmp, TDF::WORDS_PER_SAMPLE * samples + TDF::HEADER_WORDS + TDF::FLAG_WORDS));
        for (auto j=0; j<df.samples(); j++)
            h_test_out[i] += df[j].adc();
        coll.addDataFrame(DetId{(uint32_t)i}, (uint16_t*)&tmp);
    }

    cudaMalloc((void**)&d_data, size * (TDF::WORDS_PER_SAMPLE * 
        samples + TDF::HEADER_WORDS + TDF::FLAG_WORDS) * sizeof(uint16_t));
    cudaMalloc((void**)&d_out, size * sizeof(uint32_t));
    cudaMemcpy(d_data, coll.frame(0), size * (TDF::WORDS_PER_SAMPLE * 
        samples + TDF::HEADER_WORDS + TDF::FLAG_WORDS) * sizeof(uint16_t), 
            cudaMemcpyHostToDevice);
    kernel_test_hcal_qie1011_digis<TDF><<<1, size>>>(d_data, d_out, samples);
    cudaDeviceSynchronize();
    auto code = cudaGetLastError();
    if (code != cudaSuccess)
        std::cout << cudaGetErrorString(code);
    cudaMemcpy(&h_out, d_out, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // comparison
    for (auto i=0; i<size; i++) {
        std::cout << h_out[i] << " == " << h_test_out[i] << std::endl;
        assert(h_out[i] == h_test_out[i]);
    }
}



template<typename TDF>
void test_hcal_qie8_digis() {
    constexpr int n = 10;
    edm::SortedCollection<TDF> coll{n};
    TDF *d_dfs;
    uint32_t *d_out;
    uint32_t h_out[n], h_test_out[n];
    for (auto i=0; i<n; i++) {
        TDF &df = coll[i];
        df.setSize(10);
        h_test_out[i] = 0;
        uint32_t test = 0;
        for (auto j=0; j<10; j++) {
            df.setSample(j, HcalQIESample(100));
            h_test_out[i] += df.sample(j).raw();
            test += df.sample(j).raw();
        }
    }

    cudaMalloc((void**)&d_dfs, n * sizeof(TDF));
    cudaMalloc((void**)&d_out, n * sizeof(uint32_t));
    cudaMemcpy(d_dfs, &(*coll.begin()), n * sizeof(TDF), 
        cudaMemcpyHostToDevice);
    kernel_test_hcal_qie8_digis<<<1, n>>>(d_dfs, d_out);
    cudaMemcpy(&h_out, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << "collection size = " << coll.size() << std::endl;

    // comparison
    for (auto i=0; i<n; i++) {
        std::cout << h_out[i] << " == " << h_test_out[i] << std::endl;
        assert(h_out[i] == h_test_out[i]);
    }
}

void test_hcal_qie8_hbhedf() {
    HBHEDataFrame h_df, h_test_df;
    HBHEDataFrame *d_df;

    h_test_df.setSize(10);
    for (auto i=0; i<10; i++)
        h_test_df.setSample(i, HcalQIESample(100));
    h_test_df.setReadoutIds(HcalElectronicsId(100));

    cudaMalloc((void**)&d_df, sizeof(HBHEDataFrame));
    cudaMemcpy(d_df, &h_df, sizeof(HBHEDataFrame), cudaMemcpyHostToDevice);
    kernel_test_hcal_qie8_hbhedf<<<1,1>>>(d_df);
    cudaMemcpy(&h_df, d_df, sizeof(HBHEDataFrame), cudaMemcpyDeviceToHost);

    assert(h_df.size() == h_test_df.size());
    assert(h_df.elecId() == h_test_df.elecId());
    for (auto i=0; i<10; i++)
        assert(h_df[i].raw() == h_test_df[i].raw());
}

int main(int argc, char** argv) {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;

    if (nDevices > 0) {
        // qie8
        test_hcal_qiesample();
        test_hcal_qie8_hbhedf();
        test_hcal_qie8_digis<HBHEDataFrame>();
        test_hcal_qie8_digis<HFDataFrame>();
        test_hcal_qie8_digis<HODataFrame>();

        // qie1011
        test_hcal_qie1011_digis<QIE10DataFrame>();
        test_hcal_qie1011_digis<QIE11DataFrame>();
    }

    return 0;
}
