#ifndef IC_HH
#define IC_HH

//
// Federico Ferri, CEA-Saclay Irfu/SPP, 14.12.2011
// federico.ferri@cern.ch
//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TTree.h"


class IC {
        public:
                enum EcalPart { kAll, kEB, kEE };

                IC();
                //IC(const IC &);
                //IC(const EcalIntercalibConstants & ic, const EcalIntercalibErrors & eic);

                EcalIntercalibConstants & ic() { return _ic; }
                EcalIntercalibErrors & eic() { return _eic; }

                const EcalIntercalibConstants & ic() const { return _ic; }
                const EcalIntercalibErrors & eic() const { return _eic; }
                const std::vector<DetId> & ids() const { return _detId; }

                // selectors
                static bool all(DetId id);
                static bool isBorderNeighbour(DetId id);
                static bool isDeadNeighbour(DetId id, EcalChannelStatus & chStatus);
                static bool isBarrel(DetId id);
                static bool isEndcap(DetId id);
                static bool isEndcapPlus(DetId id);
                static bool isEndcapMinus(DetId id);
                static bool isNextToBoundaryEB(DetId id);
                static bool isNextToProblematicEB(DetId id);
                static bool isNextToProblematicEE(DetId id);
                static bool isNextToProblematicEEPlus(DetId id);
                static bool isNextToProblematicEEMinus(DetId id);

                // plotters
                static void constantMap(const IC & a, TH2F * h, bool (*selector)(DetId id));
                static void constantDistribution(const IC & a, TH1F * h, bool (*selector)(DetId id));
                static void profileEta(const IC & a, TProfile * h, bool (*selector)(DetId id));
                static void profilePhi(const IC & a, TProfile * h, bool (*selector)(DetId id));
                static void profileSM(const IC & a, TProfile * h, bool (*selector)(DetId id));

                // IC manipulation
                static void reciprocal(const IC & a, IC & res);
                static void multiply(const IC & a, float c, IC & res);
                static void multiply(const IC & a, const IC & b, IC & res);
                static void add(const IC & a, const IC & b, IC & res);
                static void combine(const IC & a, const IC & b, IC & res);
                static void smear(const IC & a, float sigma, IC & res);

                // tools
                static void dump(const IC & a, const char * fileName, bool (*selector)(DetId id));
                static void readSimpleTextFile(const char * fileName, IC & ic);
                static void readEcalChannelStatusFromTextFile(const char * fileName);
                static void makeRootTree(TTree & t, const IC & ic);

                static double average(const IC & a, bool (*selector)(DetId id));

        private:
                EcalIntercalibConstants   _ic;
                EcalIntercalibErrors     _eic;
                static std::vector<DetId>     _detId;
                static EcalChannelStatus channelStatus;
};

#endif
