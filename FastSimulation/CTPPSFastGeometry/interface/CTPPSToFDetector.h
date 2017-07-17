#ifndef CTPPSToFDetector_h
#define CTPPSToFDetector_h
#include <vector>
#include <map>
#include <iterator>
#include <iostream>

class CTPPSToFDetector {
      public:
            CTPPSToFDetector(int ncellx,int ncelly, std::vector<double>& cellw,double cellh,double pitchx,double pitchy,double pos, int res);
            CTPPSToFDetector(int ncellx,int ncelly, double cellw,double cellh,double pitchx,double pitchy,double pos, int res);
            virtual ~CTPPSToFDetector() {};

            double getHeight()                       {return detH_;};
            double getWidth()                        {return detW_;};
            double getPosition()                     {return detPosition_;};
            int findCellId(double x, double y);   // return the cell id corresponding to the given position
            bool get_CellCenter(int cell_id, double& x, double& y); // return false if it failed
            int get_CellId(int idx)                  {if (idx>=(int)theToFInfo.size()) return 0;
                                                      std::map<int,std::vector<double> >::const_iterator it=theToFInfo.begin();
                                                      std::advance(it,idx);
                                                      return it->first;
                                                     }
            int get_CellMultiplicity()               {return (int)theToFInfo.size();}; // return the number of cells with hit
            int getMultiplicityByCell(int cellid)    {if (!theToFInfo.count(cellid)) return 0;
                                                      return (int)theToFInfo.at(cellid).size();}; // return the hit multiplicity of the given cell

            int get_nHits_()                          {return nHits_;}; // return the total hit multiplicity (full det)
            std::vector<double> get_ToF(int cell)    {if (!theToFInfo.count(cell)) return std::vector<double>();
                                                      return theToFInfo.at(cell);
                                                     }; // return the tof of the given cell
            int getADC(int cell, int hit)            {
                                                      if (!nADC_.count(cell)) return 0;
                                                      if ((int)nADC_.at(cell).size()<hit) return 0;
                                                      return nADC_.at(cell).at(hit);
                                                     }


            void AddHit(double x, double y,double tof);
            void clear() {detId_=0;nHits_=0;theToFInfo.clear();};
      private:
            int                 nCellX_;
            int                 nCellY_;
            double              cellWq_; // width (X, horizontal dimension in mm) 
            std::vector<double> cellW_;//move to vector - diamond geometry
            double              cellH_; // height(Y, vertical dimension in mm)
            double              pitchX_; // distance (in X) between cells
            double              pitchY_; // distance (in Y) between cells
            int                 fToFResolution_;  // in ps
            std::vector<std::pair<double,double> > cellColumn_; // lower and upper limits of cells in X
            std::vector<std::pair<double,double> > cellRow_;    // lower and upper limits of cells in Y
//
            double              detW_; // detector width
            double              detH_; // detector height
            double              detPosition_; // detector position from beam (absolute value)
//
            int                 detId_;
            int                 nHits_;
            std::map<int,std::vector<int> >    nADC_;   // fake ADC counter: in case of multiple hits in the same cell, 
                                                    // it  counts the number of overlaps
            std::map<int,std::vector<double> > theToFInfo;

            typedef std::map<int,std::vector<double> > theToFInfo_t; // define a type for the tof info
};
#endif
