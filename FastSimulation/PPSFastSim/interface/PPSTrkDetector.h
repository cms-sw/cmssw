#ifndef PPSTrkDetector_h
#define PPSTrkDetector_h
#include <vector>

class PPSTrkDetector {
      public:
            PPSTrkDetector(double detw, double deth, double detin);
            virtual ~PPSTrkDetector() {};

      public:
            const double        DetectorWidth;
            const double        DetectorHeight;
            const double        DetectorPosition;
            int                 DetId;
            int                 NHits;
            std::vector<double> X;
            std::vector<double> Y;
            std::vector<double> Z;
            void clear() {DetId=0;NHits=0;X.clear();Y.clear();Z.clear();};
            void AddHit(double x, double y, double z);
};

typedef std::pair<PPSTrkDetector,PPSTrkDetector> PPSTrkStation;

#endif
