#ifndef DataFormats_HepMCCandidate_PdfInfo_h
#define DataFormats_HepMCCandidate_PdfInfo_h
/** \class reco::PdfInfo
 *
 * \verson $Id$
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
