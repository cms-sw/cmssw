#ifndef RecoJets_FFTJetAlgorithms_LookupTable2d_h
#define RecoJets_FFTJetAlgorithms_LookupTable2d_h

#include <vector>

namespace fftjetcms {
    class LookupTable2d
    {
    public:
        LookupTable2d(unsigned nx, double xmin, double xmax,
                      unsigned ny, double ymin, double ymax,
                      const std::vector<double>& data);

        inline const std::vector<double>& data() const {return data_;}
        inline unsigned nx() const {return nx_;}
        inline unsigned ny() const {return ny_;}
        inline double xmin() const {return xmin_;}
        inline double xmax() const {return xmax_;}
        inline double ymin() const {return ymin_;}
        inline double ymax() const {return ymax_;}
        inline double xstep() const {return bwx_;}
        inline double ystep() const {return bwy_;}
        inline double binValue(const unsigned ix, const unsigned iy) const
            {return data_.at(ix*ny_ + iy);}

        double closest(double x, double y) const;

    private:
        LookupTable2d();

        std::vector<double> data_;
        unsigned nx_;
        unsigned ny_;
        double xmin_;
        double xmax_;
        double ymin_;
        double ymax_;
        double bwx_;
        double bwy_;
    };
}

#endif // RecoJets_FFTJetAlgorithms_LookupTable2d_h
