#ifndef CTPPSTrkDetector_h
#define CTPPSTrkDetector_h
#include <vector>

class CTPPSTrkDetector {
      public:
            CTPPSTrkDetector(double detw, double deth, double detin);
            virtual ~CTPPSTrkDetector() {};

      public:
            const double        DetectorWidth;
            const double        DetectorHeight;
            const double        DetectorPosition;
            std::vector<unsigned int>            DetId;
            int                 NHits;
            std::vector<double> X;
            std::vector<double> Y;
            std::vector<double> Z;
            void clear() {DetId.clear();NHits=0;X.clear();Y.clear();Z.clear();};
            void AddHit(unsigned int detID,double x, double y, double z);
};

typedef std::pair<CTPPSTrkDetector,CTPPSTrkDetector> CTPPSTrkStation;

#endif
