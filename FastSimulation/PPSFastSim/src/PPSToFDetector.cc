#include "FastSimulation/PPSFastSim/interface/PPSToFDetector.h"
#include <math.h>
PPSToFDetector::PPSToFDetector(int ncellx,int ncelly, double cellw,double cellh,double pitchx,double pitchy,double pos, int res):
        NCellX(ncellx),NCellY(ncelly),CellW(cellw),CellH(cellh),PitchX(pitchx),PitchY(pitchy),fToFResolution(res),DetPosition(pos) {
//
     DetW=NCellX*CellW+(NCellX-1)*PitchX;
     DetH=NCellY*CellH+(NCellY-1)*PitchY;
// the vertical positions starts from the negative(bottom) to the positive(top) corner
     for(int i=0;i<NCellY;i++) {
// vector index points to the row number from below
        double y1=CellH*(i-NCellY/2.)+PitchY*(i-(NCellY-1)/2.);
        double y2=y1+CellH;
        CellRow.push_back(std::pair<double,double>(y1,y2));
     }
// vector index points to the column number
     for(int i=0;i<NCellX;i++) {
        double x1 = -(CellW*i+PitchX*i);
        x1-=DetPosition; // shift the limit of a column depending on the detector position
        double x2 = x1-CellW;
        CellColumn.push_back(std::pair<double,double>(x1,x2));
     }
};
void PPSToFDetector::AddHit(double x, double y, double tof) {
   int cellid = findCellId(x,y);
   if (cellid==0) return;
   if (ToFInfo.find(cellid)==ToFInfo.end()) ToFInfo[cellid]; // add empty cell
   std::vector<double>* tofs = &(ToFInfo.find(cellid)->second);
   int ntof = tofs->size();
   int i=0;
   for(;i<ntof;i++) {
      if (fabs(tofs->at(i)-tof)/fToFResolution<3) {
         tofs->at(i)=(tofs->at(i)+tof)/2.;
         nADC.at(cellid).at(i)++;
         return;
      }
   }
   tofs->push_back(tof); // no other ToF inside resolution found
   NHits++;
   nADC[cellid].push_back(1);
}
int PPSToFDetector::findCellId(double x, double y)
{
   int y_idx,x_idx;
// first, get the row number
   unsigned int i;
   unsigned int start_idx=0;
   unsigned int end_idx=CellRow.size();
   for(i=0;i<CellRow.size();i++){
      if (y>=CellRow.at(i).first&&y<=CellRow.at(i).second) break;
   } 
   if (i>=CellRow.size()) return 0;
   y_idx = i+1;
   start_idx=0;end_idx=CellColumn.size();
   for(i=start_idx;i<end_idx;i++) {
      if (x<=CellColumn.at(i).first&&x>CellColumn.at(i).second) break;
   }
   if (i>=end_idx) return 0;
   x_idx=i+1-start_idx;
   return 100*y_idx+x_idx;
}
bool PPSToFDetector::get_CellCenter(int cell_id, double& x, double& y)
{
   if (cell_id==0) return false;
   //if(!isValidCellId(cell_id)) return 0;
   unsigned int y_idx=int(cell_id/100);
   unsigned int x_idx=cell_id-y_idx*100;
   x = (CellColumn.at(x_idx-1).first+CellColumn.at(x_idx-1).second)/2.0;
   y = (CellRow.at(y_idx-1).first+CellRow.at(y_idx-1).second)/2.0;
   return true;
}
