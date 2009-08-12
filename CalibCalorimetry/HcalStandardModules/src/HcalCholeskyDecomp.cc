// Original Author:  Annabel Downs
//         Created:  Wed Jul 29 10:54:11 CEST 2009
#include <memory>
#include "CalibCalorimetry/HcalStandardModules/interface/HcalCholeskyDecomp.h"

HcalCholeskyDecomp::HcalCholeskyDecomp(const edm::ParameterSet& iConfig)
{
  outfile = iConfig.getUntrackedParameter<std::string>("outFile","CholeskyMatrices.txt");
}


HcalCholeskyDecomp::~HcalCholeskyDecomp()
{
}

void
HcalCholeskyDecomp::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   edm::ESHandle<HcalCovarianceMatrices> refCov;
   iSetup.get<HcalCovarianceMatricesRcd>().get("reference",refCov);
   const HcalCovarianceMatrices* myCov = refCov.product(); //Fill emap from database

   double sig[4][10][10];
   double c[4][10][10], cikik[4], cikjk[4];

   HcalCholeskyMatrices * outMatrices = new HcalCholeskyMatrices();

   std::vector<DetId> listChan = myCov->getAllChannels();
   std::vector<DetId>::iterator cell;
   for (std::vector<DetId>::iterator it = listChan.begin(); it != listChan.end(); it++)
   {
   for(int m= 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = 0; j!=10; j++){
            sig[m][i][j] = 0;
            c[m][i][j] = 0;
            cikik[m] =0;
            cikjk[m] = 0;
         }
      }
   }

   const HcalCovarianceMatrix * CMatrix = myCov->getValues(*it);

   for(int m = 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = 0; j != 10; j++) {sig[m][i][j] = CMatrix->getValue(m,i,j);}
      }
   }

   for(int m = 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = i; j != 10; j++) sig[m][j][i] = sig[m][i][j];
      }
   }
   //.......................Cholesky Decomposition.............................
   //Step 1a
   for(int m = 0; m!=4; m++) c[m][0][0]=sqrt(sig[m][0][0]);
   for(int m = 0; m!=4; m++)
      for(int i = 1; i != 10; i++)
         c[m][i][0] = sig[m][0][i] / c[m][0][0];
   //Chelesky Matrices
   for(int m = 0; m!=4; m++){
      c[m][1][1] = sqrt((sig[m][1][1]) - (c[m][1][0]*c[m][1][0])); // i) step 2a of chelesky decomp
      if (((sig[m][1][1]) - (c[m][1][0]*c[m][1][0]))<=0) continue;
      for(int i = 2; i !=10; i++){
         for(int j=1; j!= i; j++){
            //cikjk[m] = 0;
            for(int k=0; k != (j-1); k++)
               cikjk[m] += c[m][i][k]*c[m][j][k]; // ii)  step 2a of chelesky decomp
            c[m][i][j] = (sig[m][i][j] - cikjk[m])/c[m][j][j]; // step 3a of chelesky decomp
         }
         // cikik[m] = 0;
         for( int k = 0; k != (i-1); k++)
                                 cikik[m] += c[m][i][k]*c[m][i][k];
         double test = ((sig[m][i][i]) - cikik[m]);
         if(test > 0 ){c[m][i][i] = sqrt(test);}
         else{
           c[m][i][i] = -.001234;
         }
     }
 
   } 

   HcalCholeskyMatrix * item = new HcalCholeskyMatrix(it->rawId());
   for(int m = 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = i; j != 10; j++) item->setValue(m,j,i,sig[m][i][j]);
      }
   }
   
   outMatrices->addValues(*item);

   }

   ofstream cmatout(outfile.c_str());
   HcalDbASCIIIO::dumpObject(cmatout, *outMatrices);
}

// ------------ meth(d called once each job just before starting event loop  ------------
void 
HcalCholeskyDecomp::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalCholeskyDecomp::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalCholeskyDecomp);
