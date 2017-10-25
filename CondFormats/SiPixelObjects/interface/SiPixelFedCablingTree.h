#ifndef SiPixelFedCablingTree_H
#define SiPixelFedCablingTree_H

#include <vector>
#include <unordered_map>
#include <string>

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

class SiPixelFedCablingTree final : public  SiPixelFedCabling {

public:
  typedef sipixelobjects::PixelFEDCabling PixelFEDCabling;

  SiPixelFedCablingTree(const std::string & version="") : theVersion(version) {}

  ~SiPixelFedCablingTree() override {}

  /// add cabling for one fed
  void addFed(const PixelFEDCabling& f);

  /// get fed identified by its id
  const PixelFEDCabling * fed(unsigned int idFed) const;

  std::vector<const PixelFEDCabling *> fedList() const;

  ///map version
  std::string version() const override { return theVersion; }

  std::string print(int depth = 0) const;

  void addItem(unsigned int fedId, unsigned int linkId, const sipixelobjects::PixelROC& roc);

  std::vector<sipixelobjects::CablingPathToDetUnit> pathToDetUnit(uint32_t rawDetId) const override;

  const sipixelobjects::PixelROC* findItem(const sipixelobjects::CablingPathToDetUnit & path) const override;  

  const sipixelobjects::PixelROC* findItemInFed(const sipixelobjects::CablingPathToDetUnit & path, 
						const PixelFEDCabling * aFed) const;  


  std::unordered_map<uint32_t, unsigned int> det2fedMap() const override;
  std::map< uint32_t,std::vector<sipixelobjects::CablingPathToDetUnit> > det2PathMap() const override;


  int checkNumbering() const;

private:
  std::string theVersion; 
  std::unordered_map<int, PixelFEDCabling> theFedCablings;
};
#endif
