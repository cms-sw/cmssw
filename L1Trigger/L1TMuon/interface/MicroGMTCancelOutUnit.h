#ifndef __l1microgmtcanceloutunit_h
#define __l1microgmtcanceloutunit_h

#include "MicroGMTConfiguration.h"
#include "MicroGMTMatchQualLUT.h"

#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTLUTFactories.h"

namespace l1t {
  enum cancelmode {
    tracks, coordinate
  };

  class MicroGMTCancelOutUnit {
    public:
      MicroGMTCancelOutUnit ();
      virtual ~MicroGMTCancelOutUnit ();

      /// Initialisation from ES record
      void initialise(L1TMuonGlobalParamsHelper*);
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

      std::shared_ptr<MicroGMTMatchQualLUT> m_boPosMatchQualLUT;
      std::shared_ptr<MicroGMTMatchQualLUT> m_boNegMatchQualLUT;
      std::shared_ptr<MicroGMTMatchQualLUT> m_foPosMatchQualLUT;
      std::shared_ptr<MicroGMTMatchQualLUT> m_foNegMatchQualLUT;
      std::shared_ptr<MicroGMTMatchQualLUT> m_brlSingleMatchQualLUT;
      std::shared_ptr<MicroGMTMatchQualLUT> m_ovlPosSingleMatchQualLUT;
      std::shared_ptr<MicroGMTMatchQualLUT> m_ovlNegSingleMatchQualLUT;
      std::shared_ptr<MicroGMTMatchQualLUT> m_fwdPosSingleMatchQualLUT;
      std::shared_ptr<MicroGMTMatchQualLUT> m_fwdNegSingleMatchQualLUT;
      std::map<int, std::shared_ptr<MicroGMTMatchQualLUT>> m_lutDict;
  };
}
#endif /* defined(__l1microgmtcanceloutunit_h) */
