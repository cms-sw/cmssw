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

   edm::ESHandle<HcalDbService> conditions;
   iSetup.get<HcalDbRecord>().get(conditions);

//   const HcalQIEShape* shape = conditions->getHcalShape();

   edm::ESHandle<HcalCovarianceMatrices> refCov;
   iSetup.get<HcalCovarianceMatricesRcd>().get("reference",refCov);
   const HcalCovarianceMatrices* myCov = refCov.product(); //Fill emap from database

   double sig[4][10][10];
   double c[4][10][10], cikik[4], cikjk[4];

   HcalCholeskyMatrices * outMatrices = new HcalCholeskyMatrices();

   int HBcount = 0;
   int HEcount = 0;
   int HFcount = 0;
   int HOcount = 0;

   double HBmatrix[4][10][10] = {{{0.}}};
   double HEmatrix[4][10][10] = {{{0.}}};
   double HFmatrix[4][10][10] = {{{0.}}};
   double HOmatrix[4][10][10] = {{{0.}}};


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
//   const HcalQIECoder* coder = conditions->getHcalCoder(it->rawId());

   double tempmatrix[4][10][10] = {{{0.}}};

   for(int m = 0; m != 4; m++){
      for(int i = 0; i != 10; i++){
 //          for(int j = i; j != 10; j++) item->setValue(m,(i*(i-1)/2+j)-1,sig[m][i][j]);
        for(int j = i; j != 10; j++) 
        {
           item->setValue(m,j,i, c[m][i][j]);
           tempmatrix[m][i][j] = c[m][i][j];
        }           
/*         for(int j = i; j != 10; j++){
            double x = sig[m][i][j];
            int x1=(int)std::floor(x);
            int x2=(int)std::floor(x+1);
            float y2=coder->charge(*shape,x2,m);
            float y1=coder->charge(*shape,x1,m);
            double matrixelementfc=(y2-y1)*(x-x1)+y1;
            item->setValue(m,j,i,matrixelementfc); 
         }*/
      }
   }
   
   HcalDetId hcalid(it->rawId());
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

   }  //end of loop over all covariance matrices
   
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
              std::cout << "Cholesky Matrix not found for cell " <<  HcalGenericDetId(it->rawId());
              if(it->isHcalDetId())
              {
                 HcalDetId hcalid2(it->rawId());
                 HcalCholeskyMatrix * item = new HcalCholeskyMatrix(it->rawId());
                 for(int m = 0; m != 4; m++)
                 {
                    for(int i = 0; i != 10; i++)
                    {
                       for(int j = 0; j != 10; j++)
                       {
                          if(hcalid2.subdet()==1) item->setValue(m,j,i,HBmatrix [m][i][j]);
                          if(hcalid2.subdet()==2) item->setValue(m,j,i,HEmatrix [m][i][j]);
                          if(hcalid2.subdet()==3) item->setValue(m,j,i,HFmatrix [m][i][j]);
                          if(hcalid2.subdet()==4) item->setValue(m,j,i,HOmatrix [m][i][j]);
                       }
                    }
                 }
                 outMatrices->addValues(*item);
                 std::cout << "... Average Matrix used";
              }
              std::cout << std::endl;
	  }
      }

  std::vector<DetId> checkResult = outMatrices->getAllChannels();
  int count = 0;
  for(std::vector<DetId>::const_iterator it = checkResult.begin(); it != checkResult.end(); it++) count ++;

  std::cout << "There are " << count << " channels with Cholesky matrices.\n";

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
