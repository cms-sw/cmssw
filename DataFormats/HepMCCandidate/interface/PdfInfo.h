#ifndef DataFormats_HepMCCandidate_PdfInfo_h
#define DataFormats_HepMCCandidate_PdfInfo_h
/** \class reco::PdfInfo
 *
 * \verson $Id: PdfInfo.h,v 1.2 2008/02/29 08:24:32 llista Exp $
 *
 */

namespace reco {
   struct PdfInfo {
       char   id1; 
       char   id2; 
       float  x1;
       float  x2;
       float  scalePDF; 
       float  pdf1;
       float  pdf2;
   };
}

#endif
