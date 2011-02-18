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

      typedef boost::shared_ptr<Alignments> AliPtr ;
      typedef std::vector<AlignTransform>   AliVec ;

      typedef AlignTransform::Translation Trl ;
      typedef AlignTransform::Rotation    Rot ;

      typedef const std::vector<double>& DVec ;

      static const unsigned int k_nA ;

      WriteESAlignments( const edm::EventSetup& eventSetup ,
			 DVec                   alphaVec   ,
			 DVec                   betaVec    ,
			 DVec                   gammaVec   ,
			 DVec                   xtranslVec ,
			 DVec                   ytranslVec ,
			 DVec                   ztranslVec  ) ;

      ~WriteESAlignments() ;

   private:

      void convert( const edm::EventSetup& eS ,
		    DVec                   a  ,
		    DVec                   b  ,
		    DVec                   g  ,
		    DVec                   x  ,
		    DVec                   y  ,
		    DVec                   z  ,
		    AliVec&                va  ) ;

      void write( AliPtr aliPtr ) ;
};

#endif
