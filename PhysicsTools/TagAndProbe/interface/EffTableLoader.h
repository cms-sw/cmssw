#ifndef EffTableLoader_h
#define EffTableLoader_h

#include <string>
#include <vector>
#include <utility>

class EffTableReader;
class EffTableLoader {
 public:
  EffTableLoader ();
  EffTableLoader (const std::string& fDataFile);
  virtual ~EffTableLoader ();
  std::vector<float> correctionEff (float fEt, float fEta) const; 
  std::vector<float> correctionEff (int index) const;
  int GetBandIndex(float fEt, float fEta) const;
  std::vector<std::pair<float, float> > GetCellInfo(int index)const;
  std::vector<std::pair<float, float> > GetCellInfo(float fEt, float fEta)const;
  std::pair<float, float> GetCellCenter(int index )const;
  std::pair<float, float> GetCellCenter(float fEt, float fEta )const;
  int size(void);

 private:
   EffTableReader* mParameters;
};

#endif
