#include <vector>
#include <algorithm>
struct RooAbsData;

class QuantileCalculator {
    public:
        enum Method { Simple, Sectioning, Jacknife };

        QuantileCalculator();
        ~QuantileCalculator();
        QuantileCalculator(const std::vector<double> &values, const std::vector<double> &weights = std::vector<double>());
        QuantileCalculator(const std::vector<float> &values, const std::vector<float> &weights = std::vector<float>());
        QuantileCalculator(const RooAbsData &data, const char *varName, int firstEntry=0, int lastEntry=-1);
        /// Randomize points before sectioning
        void randomizePoints() ;
        std::pair<double,double> quantileAndError(double quantile, Method method);
    private:
        struct point { 
            float x, w; 
            int set; 
            inline bool operator<(const point &other) const { return x < other.x; }
        };
        std::vector<point> points_;
        std::vector<double> sumw_;
        std::vector<float> quantiles_;

        int guessPartitions(int size, double quantile) ;
        template<typename T> void import(const std::vector<T> &values, const std::vector<T> &weights) ;
        void partition(int m, bool doJacknife) ;
        void quantiles(double quantile, bool doJacknife);
         
};

