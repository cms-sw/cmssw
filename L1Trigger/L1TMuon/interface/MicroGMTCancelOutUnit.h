#ifndef __l1microgmtcanceloutunit_h
#define __l1microgmtcanceloutunit_h

#include "MicroGMTConfiguration.h"
#include "MicroGMTMatchQualLUT.h"

namespace l1t {
  enum cancelmode {
    tracks, coordinate
  };

  class MicroGMTCancelOutUnit {
    public:
      explicit MicroGMTCancelOutUnit (const edm::ParameterSet&);
      virtual ~MicroGMTCancelOutUnit ();
      /// Cancel out between sectors/wedges in one track finder
      void setCancelOutBits(GMTInternalWedges&, tftype, cancelmode);
      /// Cancel-out between overlap and barrel track finders
      void setCancelOutBitsOverlapBarrel(GMTInternalWedges&, GMTInternalWedges&, cancelmode);
      /// Cancel-out between overlap and endcap track finders
      void setCancelOutBitsOverlapEndcap(GMTInternalWedges&, GMTInternalWedges&, cancelmode);
    private:
      /// Compares all muons from coll1 with all muons from coll2 and sets the cancel-bits based on eta/phi coordinates
      void getCoordinateCancelBits(std::vector<std::shared_ptr<GMTInternalMuon>>&, std::vector<std::shared_ptr<GMTInternalMuon>>&);
      /// Compares all muons from coll1 with all muons from coll2 and sets the cancel-bits based on track addresses
      void getTrackAddrCancelBits(std::vector<std::shared_ptr<GMTInternalMuon>>&, std::vector<std::shared_ptr<GMTInternalMuon>>&);

      MicroGMTMatchQualLUT m_boPosMatchQualLUT;
      MicroGMTMatchQualLUT m_boNegMatchQualLUT;
      MicroGMTMatchQualLUT m_foPosMatchQualLUT;
      MicroGMTMatchQualLUT m_foNegMatchQualLUT;
      MicroGMTMatchQualLUT m_brlSingleMatchQualLUT;
      MicroGMTMatchQualLUT m_ovlPosSingleMatchQualLUT;
      MicroGMTMatchQualLUT m_ovlNegSingleMatchQualLUT;
      MicroGMTMatchQualLUT m_fwdPosSingleMatchQualLUT;
      MicroGMTMatchQualLUT m_fwdNegSingleMatchQualLUT;
      std::map<int, MicroGMTMatchQualLUT*> m_lutDict;
  };
}
#endif /* defined(__l1microgmtcanceloutunit_h) */
