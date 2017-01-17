#include "DataFormats/PatCandidates/interface/liblogintpack.h"
#include "DataFormats/PatCandidates/interface/libminifloat.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

using namespace  logintpack;

pat::PackedCandidate::PackedCovariance legacyCovarianceUnpacking(  uint16_t packedCovarianceDxyDxy_, uint16_t packedCovarianceDxyDz_, uint16_t packedCovarianceDzDz_,
                                 int8_t packedCovarianceDlambdaDz_, int8_t packedCovarianceDphiDxy_,
                                 int8_t packedCovarianceDptDpt_, int8_t packedCovarianceDetaDeta_, int8_t packedCovarianceDphiDphi_ ) {
    pat::PackedCandidate::PackedCovariance m;
    m.dptdpt=unpack8log(packedCovarianceDptDpt_,-15,0); // move this to PackedCand /pt()/pt(); 
    m.detadeta=unpack8log(packedCovarianceDetaDeta_,-20,-5);
    m.dphidphi=unpack8log(packedCovarianceDphiDphi_,-15,0); // move this to packed cabd /pt()/pt(); 
    m.dphidxy=unpack8log(packedCovarianceDphiDxy_,-17,-4);
    m.dlambdadz=unpack8log(packedCovarianceDlambdaDz_,-17,-4);
    m.dxydxy=MiniFloatConverter::float16to32(packedCovarianceDxyDxy_)/10000.;
    m.dxydz=MiniFloatConverter::float16to32(packedCovarianceDxyDz_)/10000.;
    m.dzdz=MiniFloatConverter::float16to32(packedCovarianceDzDz_)/10000.;
   return m;
}

