#include "FastSimulation/CTPPSFastGeometry/interface/CTPPSToFDetector.h"
#include <math.h>


CTPPSToFDetector::CTPPSToFDetector(int ncellx,int ncelly, std::vector<double>& cellw,double cellh,double pitchx,double pitchy,double pos, int res):
    nCellX_(ncellx),nCellY_(ncelly),cellW_(cellw),cellH_(cellh),pitchX_(pitchx),pitchY_(pitchy),fToFResolution_(res),detPosition_(pos) {
        // the vertical positions starts from the negative(bottom) to the positive(top) corner
        // vector index points to the row number from below
        cellRow_.push_back(std::pair<double,double>(-cellH_*0.5,cellH_*0.5));
	cellColumn_.reserve(nCellX_); // vector index points to the column number
        for(int i=0;i<nCellX_;i++) {
            double x1 = 0., x2 = 0.; 
            if(i==0) {detW_ = pitchX_; x1 = -(detPosition_+detW_);}
            else x1 = -detPosition_+detW_; //detPosition_ - shift the limit of a column depending on the detector position
            x2 = x1-cellW_.at(i);
            detW_ += (x2-x1)-pitchX_; 
            cellColumn_.push_back(std::pair<double,double>(x1,x2));
        }
        //diamond geometry
        detH_=nCellY_*cellH_;
        detW_=-detW_-2*pitchX_; 
    };

CTPPSToFDetector::CTPPSToFDetector(int ncellx,int ncelly, double cellwq,double cellh,double pitchx,double pitchy,double pos, int res):
    nCellX_(ncellx),nCellY_(ncelly),cellWq_(cellwq),cellH_(cellh),pitchX_(pitchx),pitchY_(pitchy),fToFResolution_(res),detPosition_(pos) {
        //
        detW_=nCellX_*cellWq_+(nCellX_-1)*pitchX_;
        detH_=nCellY_*cellH_+(nCellY_-1)*pitchY_;
        // the vertical positions starts from the negative(bottom) to the positive(top) corner
        cellRow_.reserve(nCellY_); // vector index points to the row number from below
        for(int i=0;i<nCellY_;i++) {
            double y1=cellH_*(i-nCellY_*0.5)+pitchY_*(i-(nCellY_-1)*0.5);
            double y2=y1+cellH_;
            cellRow_.push_back(std::pair<double,double>(y1,y2));
        }
        cellColumn_.reserve(nCellX_);// vector index points to the column number
        for(int i=0;i<nCellX_;i++) {
            double x1 = -(cellWq_*i+pitchX_*i);
            x1-=detPosition_; // shift the limit of a column depending on the detector position
            double x2 = x1-cellWq_;
            cellColumn_.push_back(std::pair<double,double>(x1,x2));
        }
    };
void CTPPSToFDetector::AddHit(double x, double y, double tof) {
    int cellid = findCellId(x,y);
    if (cellid==0) return;
    if (theToFInfo.find(cellid)==theToFInfo.end()) theToFInfo[cellid]; // add empty cell
    std::vector<double>* tofs = &(theToFInfo.find(cellid)->second);
    int ntof = tofs->size();
    int i=0; double oneOverRes=1.0/fToFResolution_;
    for(;i<ntof;i++) {
        if (fabs(tofs->at(i)-tof)*oneOverRes<3) {
            tofs->at(i)=(tofs->at(i)+tof)/2.;
            nADC_.at(cellid).at(i)++;
            return;
        }
    }
    tofs->push_back(tof); // no other ToF inside resolution found
    nHits_++;
    nADC_[cellid].push_back(1);
}
int CTPPSToFDetector::findCellId(double x, double y)
{
    int y_idx,x_idx;
    // first, get the row number
    unsigned int i;
    unsigned int start_idx=0;
    unsigned int end_idx=cellRow_.size();
    for(i=0;i<cellRow_.size();i++){
        if (y>=cellRow_.at(i).first&&y<=cellRow_.at(i).second) break;
    } 
    if (i>=cellRow_.size()) return 0;
    y_idx = i+1;
    start_idx=0;end_idx=cellColumn_.size();
    for(i=start_idx;i<end_idx;i++) {
        if (x<=cellColumn_.at(i).first&&x>cellColumn_.at(i).second) break;
    }
    if (i>=end_idx) return 0;
    x_idx=i+1-start_idx;
    return 100*y_idx+x_idx;
}
bool CTPPSToFDetector::get_CellCenter(int cell_id, double& x, double& y)
{
    if (cell_id==0) return false;
    //if(!isValidCellId(cell_id)) return 0;
    unsigned int y_idx=int(cell_id*0.01);
    unsigned int x_idx=cell_id-y_idx*100;
    x = (cellColumn_.at(x_idx-1).first+cellColumn_.at(x_idx-1).second)/2.0;
    y = (cellRow_.at(y_idx-1).first+cellRow_.at(y_idx-1).second)/2.0;
    return true;
}
