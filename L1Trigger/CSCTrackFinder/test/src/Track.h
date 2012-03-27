
#ifndef jhugon_Track_h
#define jhugon_Track_h
// system include files
#include <vector>
#include <string>

#include "L1Trigger/CSCTrackFinder/test/src/TrackHistogramList.h"

namespace csctf_analysis
{
  class Track
  {
    public:
	Track();
	virtual ~Track();
	Track(const Track& track);
        virtual void print() {};
	virtual void matchedTo(int i, double newR);
        virtual void unMatch();
	virtual void setHistList(TrackHistogramList* histolist)
		{
			histlist=histolist;
		};
	virtual void fillHist();
	virtual void fillMatchHist();
	//virtual void ghostMatchedTo(int i, double newR);
	virtual bool getGhost() const {return ghost;};
	virtual void loseBestGhostCand(std::string param);
	virtual std::vector<unsigned int>* ghostMatchedIndecies()const
		{
			return ghostMatchedToIndex;
		};
	virtual void fillGhostHist();
	virtual double getPt() const {return 0;};
	virtual double getPhi() const {return 0;};
	virtual double getEta() const {return 0;};
	virtual int getQuality() const {return Quality;};
	virtual double getP() const {return 0;};
	virtual double getPz() const {return 0;};
	virtual double getRadius() const {return 0;};
	virtual bool getMatched() const {return matched;};
	virtual bool getMatchedIndex() const {return matchedIndex;};
	virtual double getR() const {return R;};
	virtual double getTFPt() const {return TFPt;};
	virtual void setTFPt(const double Pt)  {TFPt=Pt;};
	virtual void setQuality(const int Q) {Quality=Q;};
	virtual int getMode() const {return -1;};//Added by nathaniel
	virtual double getRank() const {return -1;};
	virtual int getPtPacked() const {return -1;};
	virtual int getEtaPacked() const {return -1;};
	virtual int getPhiPacked() const {return -1;};
	virtual int getChargePacked() const {return -1;};
	virtual int getFR() const {return -1;};
	virtual int getLUTAddress() const {return -1;};
	virtual void fillSimvTFHist(const Track& simtrack, const Track& tftrack) const;
	virtual void fillRateHist();

    private:
	TrackHistogramList* histlist;
    protected:
	int matchedIndex;
	double R;
	bool matched;
	int Quality;
	double TFPt;
	bool ghost;
	std::vector<unsigned int>* ghostMatchedToIndex;
	std::vector<double>* ghostR;
	std::vector<double>* ghostQ;
  };
}
#endif
