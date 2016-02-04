#ifndef SOMEPACKAGE_WRITEESALIGNMENTS_H
#define SOMEPACKAGE_WRITEESALIGNMENTS_H 1

#include "boost/shared_ptr.hpp"

namespace edm
{
   class EventSetup ;
}

#include "CondFormats/Alignment/interface/Alignments.h"


class WriteESAlignments
{
   public:

      typedef Alignments*                  AliPtr ;
      typedef std::vector<AlignTransform>  AliVec ;

      typedef AlignTransform::Translation Trl ;
      typedef AlignTransform::Rotation    Rot ;

      typedef std::vector<double> DVec ;

      static const unsigned int k_nA ;

      WriteESAlignments( const edm::EventSetup& eventSetup ,
			 const DVec&            alphaVec   ,
			 const DVec&            betaVec    ,
			 const DVec&            gammaVec   ,
			 const DVec&            xtranslVec ,
			 const DVec&            ytranslVec ,
			 const DVec&            ztranslVec  ) ;

      ~WriteESAlignments() ;

   private:

      void convert( const edm::EventSetup& eS ,
		    const DVec&            a  ,
		    const DVec&            b  ,
		    const DVec&            g  ,
		    const DVec&            x  ,
		    const DVec&            y  ,
		    const DVec&            z  ,
		    AliVec&                va  ) ;

      void write( AliPtr aliPtr ) ;
};

#endif
