#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnitMux.h"

using namespace l1t::me0;

uint64_t l1t::me0::parse_data(const UInt192& data, int strip, int max_span) {
    UInt192 data_shifted;
    uint64_t parsed_data;
    if (strip < max_span/2 + 1) {
        data_shifted = data << (max_span/2 - strip);
        parsed_data = (data_shifted & UInt192(0xffffffffffffffff >> (64 - max_span))).to_ullong();
        // parsed_data = (data_shifted & UInt192(pow(2,max_span)-1)).to_ullong();
    }
    else {
        data_shifted = data >> (strip - max_span/2);
        parsed_data = (data_shifted & UInt192(0xffffffffffffffff >> (64 - max_span))).to_ullong();
        // parsed_data = (data_shifted & UInt192(pow(2,max_span)-1)).to_ullong();
    }
    return parsed_data;
}
std::vector<uint64_t> l1t::me0::extract_data_window(const std::vector<UInt192>& ly_dat, int strip, int max_span) {
    std::vector<uint64_t> out;
    for (const UInt192& data : ly_dat) {
        out.push_back(parse_data(data,strip,max_span));
    }
    return out;
}
std::vector<int> l1t::me0::parse_bx_data(const std::vector<int>& bx_data, int strip, int max_span) {
    std::vector<int> data_shifted;
    std::vector<int> parsed_bx_data;
    if (strip < max_span/2 + 1) {
        std::vector<std::vector<int>> seed = {std::vector<int>((max_span/2 - strip),-9999), bx_data};
        data_shifted = concatVector(seed);
        parsed_bx_data = std::vector<int>(data_shifted.begin(),data_shifted.begin()+max_span);
    }
    else {
        int shift = strip - max_span / 2;
        int num_appended_nedded = shift + max_span - static_cast<int>(bx_data.size());
        if (num_appended_nedded > 0) {
            std::vector<std::vector<int>> seed = {bx_data, std::vector<int>(num_appended_nedded,-9999)};
            data_shifted = concatVector(seed);
        }
        else {
            data_shifted = bx_data;
        }
        parsed_bx_data = std::vector<int>(data_shifted.begin()+shift,data_shifted.begin()+shift+max_span);
    }
    return parsed_bx_data;
}
std::vector<std::vector<int>> l1t::me0::extract_bx_data_window(const std::vector<std::vector<int>>& ly_dat, int strip, int max_span) {
    std::vector<std::vector<int>> out;
    for (const std::vector<int>& data : ly_dat) {
        out.push_back(parse_bx_data(data,strip,max_span));
    }
    return out;
}
std::vector<ME0StubPrimitive> l1t::me0::pat_mux(const std::vector<UInt192>& partition_data,
                                                const std::vector<std::vector<int>>& partition_bx_data,
                                                int partition, Config& config) {
    std::vector<ME0StubPrimitive> out;
    for (int strip=0; strip<config.width; ++strip) {
        const std::vector<uint64_t>& data_window = extract_data_window(partition_data, strip, config.max_span);
        const std::vector<std::vector<int>>& bx_data_window = extract_bx_data_window(partition_bx_data, strip, config.max_span);
        const ME0StubPrimitive& seg = pat_unit(data_window,
                                               bx_data_window,
                                               strip,
                                               partition,
                                               config.ly_thresh_patid,
                                               config.ly_thresh_eta,
                                               config.max_span,
                                               config.skip_centroids,
                                               config.num_or);
        out.push_back(seg);
    }
    return out;
}
