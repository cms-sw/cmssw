
#ifndef jhugon_RHistogram_h
#define jhugon_RHistogram_h
// system include files
#include <vector>
#include <string>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <L1Trigger/CSCTrackFinder/test/src/Track.h>
#include <L1Trigger/CSCTrackFinder/test/src/TFTrack.h>

#include <TH1.h>
#include <TH2.h>

namespace csctf_analysis
{
  class RHistogram
  {
    public:
	RHistogram(std::string histPrefix);
	void fillR(TFTrack track1, TFTrack track2);

    private:
	std::string m_histPrefix;

	edm::Service<TFileService> m_fs;

	TH1F* m_histR;
	TH2F* m_histRvEtaHigh;
	TH2F* m_histRvEtaLow;
	TH2F* m_histRvPhiHigh;
	TH2F* m_histRvPhiLow;
	TH2F* m_histRvPtHigh;
	TH2F* m_histRvPtLow;
  };
}
#endif
