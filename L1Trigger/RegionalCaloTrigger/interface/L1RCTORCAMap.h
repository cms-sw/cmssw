#ifndef L1RCTORCAMap_h
#define L1RCTORCAMap_h

#include <vector>

class L1RCTORCAMap {

 public:
 
  L1RCTORCAMap();
  
  std::vector<std::vector<std::vector<unsigned short> > > giveBarrel();
  std::vector<std::vector<unsigned short> > giveHF();

  void readData(const std::vector<unsigned>& emet, const std::vector<unsigned>&  hdet,
		const std::vector<bool>& emfg, const std::vector<bool>& hdfg,
		const std::vector<unsigned>& hfet);

  std::vector<int> orcamap(int eta, int phi);

  unsigned short combine(unsigned short et, unsigned short fg);
  std::vector<unsigned short> combVec(const std::vector<unsigned short>& et,
				 const std::vector<unsigned short>& fg);
  
  void makeBarrelData();
  void makeHFData();

 private:

  std::vector<std::vector<std::vector<unsigned short> > > barrelData;
  std::vector<std::vector<unsigned short> > hfData;

  //the barrel data comes in big nasty strips of phi
  //increasing in eta from -3 up to 3
  //This has to be broken up into the proper tower number.
  //The following method does this

  std::vector<int> lowEtaMap(int eta, int phi);
  std::vector<int> highEtaMap(int eta, int phi);

  std::vector<unsigned short> rawEMET;
  std::vector<unsigned short> rawEMFG;
  std::vector<unsigned short> rawHDET;
  std::vector<unsigned short> rawHDFG; 
  std::vector<unsigned short> rawHFET;
  std::vector<unsigned short> combEM;
  std::vector<unsigned short> combHD;

};

#endif
