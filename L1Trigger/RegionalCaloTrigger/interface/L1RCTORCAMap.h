#ifndef L1RCTORCAMap_h
#define L1RCTORCAMap_h

#include <vector>

using std::vector;

class L1RCTORCAMap {

 public:
 
  L1RCTORCAMap();
  
  vector<vector<vector<unsigned short> > > giveBarrel();
  vector<vector<unsigned short> > giveHF();

  void readData(vector<unsigned> emet, vector<unsigned>  hdet,
		vector<bool> emfg, vector<bool> hdfg,
		vector<unsigned> hfet);

  vector<int> orcamap(int eta, int phi);

  unsigned short combine(unsigned short et, unsigned short fg);
  vector<unsigned short> combVec(vector<unsigned short> et,
				 vector<unsigned short> fg);
  
  void makeBarrelData();
  void makeHFData();

 private:

  vector<vector<vector<unsigned short> > > barrelData;
  vector<vector<unsigned short> > hfData;

  //the barrel data comes in big nasty strips of phi
  //increasing in eta from -3 up to 3
  //This has to be broken up into the proper tower number.
  //The following method does this

  vector<int> lowEtaMap(int eta, int phi);
  vector<int> highEtaMap(int eta, int phi);

  vector<unsigned short> rawEMET;
  vector<unsigned short> rawEMFG;
  vector<unsigned short> rawHDET;
  vector<unsigned short> rawHDFG; 
  vector<unsigned short> rawHFET;
  vector<unsigned short> combEM;
  vector<unsigned short> combHD;

};

#endif
