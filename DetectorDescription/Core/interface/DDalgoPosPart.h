#ifndef DDalgoPosPart_h
#define DDalgoPosPart_h


class DDLogicalPart;
class DDName;
class DDAlgo;

/*! algorithmic positioning */
void DDalgoPosPart(const DDLogicalPart & self,
                   const DDLogicalPart & parent,
		   DDAlgo & algo
		   );

/*! deprecated, use the alternatve DDalgoPosPart */
void DDalgoPosPart(const DDName & self,
                   const DDName & parent,
		   DDAlgo & algo
		   );

#endif
