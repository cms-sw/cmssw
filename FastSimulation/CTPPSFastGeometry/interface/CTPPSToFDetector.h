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

            double GetHeight()                       {return DetH;};
            double GetWidth()                        {return DetW;};
            double GetPosition()                     {return DetPosition;};
            int findCellId(double x, double y);   // return the cell id corresponding to the given position
            bool get_CellCenter(int cell_id, double& x, double& y); // return false if it failed
            int get_CellId(int idx)                  {if (idx>=(int)ToFInfo.size()) return 0;
                                                      std::map<int,std::vector<double> >::const_iterator it=ToFInfo.begin();
                                                      std::advance(it,idx);
                                                      return it->first;
                                                     }
            int get_CellMultiplicity()               {return (int)ToFInfo.size();}; // return the number of cells with hit
            int GetMultiplicityByCell(int cellid)    {if (!ToFInfo.count(cellid)) return 0;
                                                      return (int)ToFInfo.at(cellid).size();}; // return the hit multiplicity of the given cell

            int get_NHits()                          {return NHits;}; // return the total hit multiplicity (full det)
            std::vector<double> get_ToF(int cell)    {if (!ToFInfo.count(cell)) return std::vector<double>();
                                                      return ToFInfo.at(cell);
                                                     }; // return the tof of the given cell
            int GetADC(int cell, int hit)            {
                                                      if (!nADC.count(cell)) return 0;
                                                      if ((int)nADC.at(cell).size()<hit) return 0;
                                                      return nADC.at(cell).at(hit);
                                                     }


            void AddHit(double x, double y,double tof);
            void clear() {DetId=0;NHits=0;ToFInfo.clear();};
      private:
            int                 NCellX;
            int                 NCellY;
            double              CellWq; // width (X, horizontal dimension in mm) 
            std::vector<double> CellW;//move to vector - diamond geometry
            double              CellH; // height(Y, vertical dimension in mm)
            double              PitchX; // distance (in X) between cells
            double              PitchY; // distance (in Y) between cells
            int                 fToFResolution;  // in ps
            std::vector<std::pair<double,double> > CellColumn; // lower and upper limits of cells in X
            std::vector<std::pair<double,double> > CellRow;    // lower and upper limits of cells in Y
//
            double              DetW; // detector width
            double              DetH; // detector height
            double              DetPosition; // detector position from beam (absolute value)
//
            int                 DetId;
            int                 NHits;
            std::map<int,std::vector<int> >    nADC;   // fake ADC counter: in case of multiple hits in the same cell, 
                                                    // it  counts the number of overlaps
            std::map<int,std::vector<double> > ToFInfo;

            typedef std::map<int,std::vector<double> > ToFInfo_t; // define a type for the tof info
};
#endif
