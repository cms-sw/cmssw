#ifndef DDalgoPosPart_h
#define DDalgoPosPart_h

#include <DetectorDescription/Core/interface/DDPosPart.h>

class DDLogicalPart;
class DDName;
class DDAlgo;
class DDCompactView;

class DDAlgoPositioner {

 public:
  DDAlgoPositioner( DDCompactView * );
  ~DDAlgoPositioner();
  
  /*! algorithmic positioning */
  void operator()(const DDLogicalPart & self,
		  const DDLogicalPart & parent,
		  DDAlgo & algo
		  );
  
 private:
  DDCompactView * cpv_;
/* /\*! algorithmic positioning *\/ */
/* void DDalgoPosPart(const DDLogicalPart & self, */
/*                    const DDLogicalPart & parent, */
/* 		   DDAlgo & algo */
/* 		   ); */

/* /\*! deprecated, use the alternatve DDalgoPosPart *\/ */
/* void DDalgoPosPart(const DDName & self, */
/*                    const DDName & parent, */
/* 		   DDAlgo & algo */
/* 		   ); */
};
#endif
