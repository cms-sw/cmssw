#ifndef CTPPSTrkDetector_h
#define CTPPSTrkDetector_h
#include <vector>

class CTPPSTrkDetector {
      public:
            CTPPSTrkDetector(double detw, double deth, double detin);
            virtual ~CTPPSTrkDetector() {};

      public:
            const double        ppsDetectorWidth_;
            const double        ppsDetectorHeight_;
            const double        ppsDetectorPosition_;
            std::vector<unsigned int>            ppsDetId_;
            int                 ppsNHits_;
            std::vector<double> ppsX_;
            std::vector<double> ppsY_;
            std::vector<double> ppsZ_;
            void clear() {ppsDetId_.clear();ppsNHits_=0;ppsX_.clear();ppsY_.clear();ppsZ_.clear();};
            void AddHit(unsigned int detID,double x, double y, double z);
};

typedef std::pair<CTPPSTrkDetector,CTPPSTrkDetector> CTPPSTrkStation;

#endif
