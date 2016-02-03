#ifndef RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizerArray_h
#define RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterizerArray_h

#include <vector>

class Phase2TrackerClusterizerArray {

    public:

        Phase2TrackerClusterizerArray();
        Phase2TrackerClusterizerArray(unsigned int, unsigned int);
        void setSize(unsigned int, unsigned int);
        bool operator()(unsigned int, unsigned int) const;
        unsigned int rows() const;
        unsigned int columns() const;
        bool inside(unsigned int, unsigned int) const;
        void set(unsigned int, unsigned int, bool);
        unsigned int size() const;
        unsigned int index(unsigned int, unsigned int) const;

    private:

        unsigned int nrows_, ncols_;
        std::vector< bool > matrix_;

};

#endif
