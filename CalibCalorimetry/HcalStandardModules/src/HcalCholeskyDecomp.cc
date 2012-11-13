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

   HcalCholeskyMatrices * outMatrices = new HcalCholeskyMatrices(myCov->topo());

   std::vector<DetId> listChan = myCov->getAllChannels();
   std::vector<DetId>::iterator cell;

   int HBcount = 0;
   int HEcount = 0;
   int HFcount = 0;
   int HOcount = 0;

   double HBmatrix[4][10][10];
   double HEmatrix[4][10][10];
   double HFmatrix[4][10][10];
   double HOmatrix[4][10][10];

   double tempmatrix[4][10][10] ;


   for(int m= 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = 0; j!=10; j++){
          HBmatrix[m][i][j] = 0;
          HEmatrix[m][i][j] = 0;
          HFmatrix[m][i][j] = 0;
          HOmatrix[m][i][j] = 0;
         }
      }
   }

   for (std::vector<DetId>::iterator it = listChan.begin(); it != listChan.end(); it++)
   {
   for(int m= 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = 0; j!=10; j++){
            sig[m][i][j] = 0;
            c[m][i][j] = 0;
            cikik[m] =0;
            cikjk[m] = 0;
            tempmatrix[m][i][j] = 0;
         }
      }
   }

   const HcalCovarianceMatrix * CMatrix = myCov->getValues(*it);
   HcalDetId hcalid(it->rawId());

   for(int m = 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = 0; j != 10; j++) {sig[m][i][j] = CMatrix->getValue(m,i,j);}
	}
   }

//Method to check 4x10 storage
//  for(int m = 0; m != 4; m++){
//      for(int i = 0; i != 6; i++){
//         for(int j=(i+4); j!=10; j++) {sig[m][i][j] = 0;}
//        }
//   }
	

   for(int m = 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = i; j != 10; j++){ sig[m][j][i] = sig[m][i][j];}
      }
   }
   //.......................Cholesky Decomposition.............................
   //Step 1a
   for(int m = 0; m!=4; m++)
	 {
	
		c[m][0][0]=sqrt(sig[m][0][0]);
	}
  for(int m = 0; m!=4; m++)
   {
      for(int i = 1; i != 10; i++)
	{
         c[m][i][0] = (sig[m][0][i] / c[m][0][0]);
	}
   }
   
 //Chelesky Matrices
   for(int m = 0; m!=4; m++)
   {
      c[m][1][1] = sqrt(sig[m][1][1] - (c[m][1][0]*c[m][1][0])); // i) step 2a of chelesky decomp
      if (((sig[m][1][1]) - (c[m][1][0]*c[m][1][0]))<=0) continue;
      for(int i = 2; i !=10; i++)
        {
	cikik[m] = 0;
         for(int j=1; j!= i; j++)
                {
            	cikjk[m] = 0;
		    for(int k=0; k != j; k++)
                        {
                        cikjk[m] += (c[m][i][k]*c[m][j][k]); // ii)  step 2a of chelesky decomp
                        }
                c[m][i][j] = (sig[m][i][j] - cikjk[m])/c[m][j][j]; // step 3a of chelesky decomp
                }//end of j loop
         
                for( int k = 0; k != i; k++)
                {
                         cikik[m] += (c[m][i][k]*c[m][i][k]);
                }
                double test = (sig[m][i][i] - cikik[m]);
                if(test > 0 )
                {
                        c[m][i][i] = sqrt(test);
                }
                else
                {
                        c[m][i][i] = 1000;
                }
        }//end of i loop
/*
 //Cholesky Matrix for rechit (2-5 ts)
  	for (int i = 0; i!=2; i++)
	{
	  for (int j=0; j!=10; j++)
	  {
		c[m][i][j] = 0;
	  }
	}
	for (int i = 6; i!=10; i++)
        {
          for (int j=0; j!=10; j++)
          {
                c[m][i][j] = 0;
          }
        }
*/


   }//end of m loop
   

   HcalCholeskyMatrix * item = new HcalCholeskyMatrix(it->rawId());
   for(int m = 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
         for(int j = i; j != 10; j++){ 
             item->setValue(m,j,i, c[m][j][i]);
             tempmatrix[m][j][i] = c[m][j][i];
         }//sig[m][j][i]
      }
   }

   if(hcalid.subdet()==1)
   {
      for(int m = 0; m != 4; m++)
         for(int i = 0; i != 10; i++)
            for(int j = 0; j != 10; j++)
               HBmatrix[m][i][j] += tempmatrix[m][i][j];
      HBcount++;
   }
   if(hcalid.subdet()==2)
   {
      for(int m = 0; m != 4; m++)
         for(int i = 0; i != 10; i++)
            for(int j = 0; j != 10; j++)
               HEmatrix[m][i][j] += tempmatrix[m][i][j];
      HEcount++;
   }
   if(hcalid.subdet()==3)
   {
      for(int m = 0; m != 4; m++)
         for(int i = 0; i != 10; i++)
            for(int j = 0; j != 10; j++)
               HOmatrix[m][i][j] += tempmatrix[m][i][j];
      HOcount++;
   }
   if(hcalid.subdet()==4)
   {
      for(int m = 0; m != 4; m++)
         for(int i = 0; i != 10; i++)
            for(int j = 0; j != 10; j++)
               HFmatrix[m][i][j] += tempmatrix[m][i][j];
      HFcount++;
   }


   
   outMatrices->addValues(*item);

   }

   for(int m = 0; m != 4; m++)
   {
      for(int i = 0; i != 10; i++)
      {
         for(int j = 0; j != 10; j++)
         {
            HBmatrix [m][i][j] /= HBcount;
            HEmatrix [m][i][j] /= HEcount;
            HFmatrix [m][i][j] /= HFcount;
            HOmatrix [m][i][j] /= HOcount;
         }
      }
   }

  std::vector<DetId> listResult = outMatrices->getAllChannels();

  int n_avg = 0;

  edm::ESHandle<HcalElectronicsMap> refEMap;
  iSetup.get<HcalElectronicsMapRcd>().get(refEMap);
//  iSetup.get<HcalElectronicsMapRcd>().get("reference",refEMap);
  const HcalElectronicsMap* myRefEMap = refEMap.product();
  std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();

    for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
      {
      DetId mydetid = DetId(it->rawId());
        if (std::find(listResult.begin(), listResult.end(), mydetid ) == listResult.end()  )
          {
//              std::cout << "Cholesky Matrix not found for cell " <<  HcalGenericDetId(it->rawId());
              if(it->isHcalDetId())
              {
              std::cout << "Cholesky Matrix not found for cell " <<  HcalGenericDetId(it->rawId());
                 HcalDetId hcalid2(it->rawId());
                 HcalCholeskyMatrix * item = new HcalCholeskyMatrix(it->rawId());
                 for(int m = 0; m != 4; m++)
                 {
                    for(int i = 0; i != 10; i++)
                    {
                       for(int j = 0; j != 10; j++)
                       {
                          if(j <= i){
                          if(hcalid2.subdet()==1) item->setValue(m,i,j,HBmatrix [m][i][j]);
                          if(hcalid2.subdet()==2) item->setValue(m,i,j,HEmatrix [m][i][j]);
                          if(hcalid2.subdet()==3) item->setValue(m,i,j,HFmatrix [m][i][j]);
                          if(hcalid2.subdet()==4) item->setValue(m,i,j,HOmatrix [m][i][j]);
                          }else{
                          if(hcalid2.subdet()==1) item->setValue(m,i,j,0);
                          if(hcalid2.subdet()==2) item->setValue(m,i,j,0);
                          if(hcalid2.subdet()==3) item->setValue(m,i,j,0);
                          if(hcalid2.subdet()==4) item->setValue(m,i,j,0);
                          }
                       }
                    }
                 }
                 outMatrices->addValues(*item);
                 std::cout << "... Average Matrix used\n";
                 n_avg++;
              }
          }
      }

   std::vector<DetId> checkResult = outMatrices->getAllChannels();
   int count = 0;
   for(std::vector<DetId>::const_iterator it = checkResult.begin(); it != checkResult.end(); it++) count ++;

   std::cout << "There are " << count << " channels with Cholesky matrices.\n" << "Used  " << n_avg << " average values.\n";

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
