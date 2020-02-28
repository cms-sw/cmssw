#ifndef RawDataFEDSelector_h
#define RawDataFEDSelector_h

/** \class RawDataFEDSelector
 *  
 * \author M. Zanetti CERN
 */

#include "DataFormats/Common/interface/Handle.h"

#include <memory>
#include <utility>
#include <vector>

class FEDRawDataCollection;

class RawDataFEDSelector {
public:
  /// Constructor
  RawDataFEDSelector(){};

  /// Destructor
  virtual ~RawDataFEDSelector(){};

  inline void setRange(const std::pair<int, int>& range) { fedRange = range; };
  inline void setRange(const std::vector<int>& list) { fedList = list; };

  std::unique_ptr<FEDRawDataCollection> select(const edm::Handle<FEDRawDataCollection>& rawData);
  std::unique_ptr<FEDRawDataCollection> select(const edm::Handle<FEDRawDataCollection>& rawData,
                                               const std::pair<int, int>& range);
  std::unique_ptr<FEDRawDataCollection> select(const edm::Handle<FEDRawDataCollection>& rawData,
                                               const std::vector<int>& list);

private:
  std::pair<int, int> fedRange;
  std::vector<int> fedList;
};

#endif
