//
//  SiPixelTemplateReco2D.cc (Version 2.60)
//  Updated to work with the 2D template generation code
//  2.10 - Add y-lorentz drift to estimate starting point [for FPix]
//  2.10 - Remove >1 pixel requirement
//  2.20 - Fix major bug, change chi2 scan to 9x5 [YxX]
//  2.30 - Allow one pixel clusters, improve cosmetics for increased style points from judges
//  2.50 - Add variable cluster shifting to make the position parameter space more symmetric,
//         also fix potential problems with variable size input clusters and double pixel flags
//  2.55 - Fix another double pixel flag problem and a small pseudopixel problem in the edgegflagy = 3 case.
//  2.60 - Modify the algorithm to return the point with the best chi2 from the starting point scan when
//         the iterative procedure does not converge [eg 1 pixel clusters]

//
//
//  Created by Morris Swartz on 7/13/17.
//
//

#ifndef SiPixelTemplateReco2D_h
#define SiPixelTemplateReco2D_h 1

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateDefs.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate2D.h"
#else
#include "SiPixelTemplateDefs.h"
#include "SiPixelTemplate2D.h"
#endif

#include <vector>

#ifndef SiPixelTemplateClusMatrix2D
#define SiPixelTemplateClusMatrix2D 1

namespace SiPixelTemplateReco2D {
   
   struct ClusMatrix {
      float & operator()(int x, int y) { return matrix[mcol*x+y];}
      float operator()(int x, int y) const { return matrix[mcol*x+y];}
      float * matrix;
      bool * xdouble;
      bool * ydouble;
      int mrow, mcol;
   };
#endif
   
   int PixelTempReco3D(int id, float cotalpha, float cotbeta, float locBz, float locBx, int edgeflagy, int edgeflagx,
                       ClusMatrix & cluster, SiPixelTemplate2D& templ,
                       float& yrec, float& sigmay, float& xrec, float& sigmax, float& probxy, float& probQ, int& qbin, float& deltay, int& npixel);
   
   int PixelTempReco3D(int id, float cotalpha, float cotbeta, float locBz, float locBx, int edgeflagy, int edgeflagx,
                       ClusMatrix & cluster, SiPixelTemplate2D& templ,
                       float& yrec, float& sigmay, float& xrec, float& sigmax, float& probxy, float& probQ, int& qbin, float& deltay);
   
   
}

#endif
