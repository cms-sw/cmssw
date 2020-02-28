#ifndef CASTORTEXT2DETIDCONVERTER_H
#define CASTORTEXT2DETIDCONVERTER_H

/** \class CastorText2DetIdConverter

   Converts any flavour of HcalDetId to/from ascii strings

*/
#include <string>
#include "DataFormats/DetId/interface/DetId.h"

class CastorText2DetIdConverter {
public:
  CastorText2DetIdConverter(const std::string& fFlavor = "NA",
                            const std::string& fField1 = "0",
                            const std::string& fField2 = "0",
                            const std::string& fField3 = "0");

  CastorText2DetIdConverter(DetId fId);

  //  bool isHcalDetId () const;
  bool isHcalCastorDetId() const;

  const std::string& getFlavor() const { return flavorName; }
  const std::string& getField1() const { return field1; }
  const std::string& getField2() const { return field2; }
  const std::string& getField3() const { return field3; }
  DetId getId() const { return mId; }
  int getField(int i) const;
  std::string toString() const;

private:
  bool init(const std::string& fFlavor,
            const std::string& fField1,
            const std::string& fField2,
            const std::string& fField3);
  bool init(DetId fId);
  void setField(int i, int fValue);
  std::string flavorName;
  std::string field1;
  std::string field2;
  std::string field3;
  DetId mId;
};

#endif
