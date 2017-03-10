#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#include "TFile.h"
#include "TH2F.h"

struct Point {
    float x[2]; // position
    int w;      // point weight
    int n;      // cluster index
};

struct Center {
    float x[2]; // position
    int n;      // cluster index
    float area, lim[2][2];
};

class ClusterAnalyzer {
public:
    static constexpr double Loss = 0.005; // Total loss, loss in one direction is Loss/2
    static constexpr size_t MinEntries = 100;
    static constexpr int exMax = 10, eyMax = 15;
    
    
    void analyze(const std::string& inputFileName, const std::string& outputFileName,
                 const std::string& mapFileName = "", bool print = false)
    {
        static const std::vector<std::string> subName = { "barrel", "endcap" };
        
        TFile resFile(inputFileName.c_str(), "READ");
        std::ofstream outFile(outputFileName);
        std::shared_ptr<std::ofstream> mapFile;
        if(mapFileName.size())
            mapFile = std::make_shared<std::ofstream>(mapFileName);
        std::ostringstream flag;
        
        for(size_t is = 0; is < subName.size(); is++) {
            for(int ix=  0; ix<= exMax; ix++) {
                std::cerr << " " << subName.at(is) << " dx=" << ix << " ";
                
                for(int iy=0; iy<=eyMax; iy++) {
                    std::ostringstream histName;
                    histName << "hspc_" << is << "_" << ix << "_" << iy;
                    TH2F * histo = (TH2F *) resFile.Get(histName.str().c_str());
                    
                    std::cerr << ".";
                    
                    // Initial guess
                    std::vector<Center> c(2);
                    c[0].n = 0;
                    c[0].x[0] = ix+1;
                    c[0].x[1] = iy+1;
                    
                    c[1].n = 1;
                    c[1].x[0] =-ix-1;
                    c[1].x[1] =-iy-1;
                    
                    flag << subName.at(is) << "_" << ix << "_" << iy;
                    
                    process(histo,c, flag.str(), print);
                    
                    if(histo->Integral() >= MinEntries) {
                        // Fix barrel_0_0
                        if(is==0 && ix==0 && iy==0) {
                            c[0].lim[1][0] = 0.;
                            c[1].lim[1][1] = 0.;
                        }
                        
                        outFile << is << " " << ix << " " << iy;
                        
                        for(int b=0; b<2; b++) {
                            for(int d=0; d<2; d++) {
                                outFile << " " << c[b].lim[d][0] << " " << c[b].lim[d][1];
                            }
                        }
                        
                        for(int b=0; b<2; b++) {
                            if(c[b].area > 0 )
                                outFile << " " << c[b].n / c[b].area;
                            else
                                outFile << " " << 0.;
                            
                            outFile << " " << c[b].n;
                        }
                        
                        outFile << std::endl;
                    }

                    if(mapFile)
                        *mapFile << ix << " " << iy << " " << histo->Integral() << std::endl;
                }
                outFile << std::endl;
                
                if(mapFile)
                    *mapFile << ix << " " << eyMax+1 << " " << 0 << "\n" << std::endl;
                
                std::cerr << std::endl;
            }

            if(mapFile) {
                for(int iy=0; iy<=eyMax+1; iy++)
                    *mapFile << exMax+1 << " " << iy << " " << 0 << std::endl;
                *mapFile << std::endl << std::endl;
            }
            
            outFile << std::endl;
        }
    }
    
private:
    static float distance(const Center& c, const Point& p)
    {
        float x[2];
        
        for(int d=0; d<2; d++)
            x[d] = std::fabs(p.x[d] - c.x[d]);
        
        return x[0]*x[0] + x[1]*x[1];
    }

    void getVectors(const std::vector<Point>& points, TH1D * x[2][2]) const
    {
        for(int n = 0 ; n < 2; n++)
            for(int d = 0 ; d < 2; d++)
            {
                std::ostringstream name;
                name << n << "_" << d;
                x[n][d] = (TH1D *) line->Clone(name.str().c_str());
            }
        
        for(const auto& point : points) {
            for(int d=0; d<2; d++)
                x[point.n][d]->Fill(point.x[d], point.w);
        }
    }
    
    static double getFraction(const TH1D * line, double fraction)
    {
        double n = line->Integral();
        double s = 0;
        
        for(int i = 0 ; i <= line->GetNbinsX()+1; i++)
        {
            s += line->GetBinContent(i);
            if(s > n * fraction)
            {
                return line->GetXaxis()->GetBinUpEdge(i) -
                (s - n * fraction) / line->GetBinContent(i) *
                line->GetXaxis()->GetBinWidth(i);
            }
        }
        
        std::cerr << " exit!" << std::endl;
        return 0.;
    }
    
    void kMeans(std::vector<Point>& points, std::vector<Center>& c) const
    {
        int changes;
        do {
            changes = 0;
            
            // Sort point into clusters
            for(auto& point : points) {
                int n;
                if(distance(c[0], point) < distance(c[1], point)) n = 0; else n = 1;
                
                if(n != point.n)
                { point.n = n; changes++; }
            }
            
            if(changes != 0)
            {
                // Re-compute centers
                TH1D * x[2][2];
                getVectors(points, x);
                
                for(int n=0; n<2; n++) // which cluster
                    for(int d=0; d<2; d++) // which direction
                    {
                        c[n].n = int(x[n][d]->Integral() + 0.5);
                        
                        if(x[n][d]->Integral() > 0)
                            c[n].x[d] = getFraction(x[n][d], 0.5);
                    }
            }
        } while(changes != 0);
    }

    void findLimits(const std::vector<Point>& points, std::vector<Center>& c) const
    {
        TH1D * x[2][2];
        getVectors(points, x);
        
        for(int b=0; b<2; b++) // branch
        {
            for(int d=0; d<2; d++) // direction
                if(x[b][d]->Integral() > 0)
                {
                    double limits[2] = {0, 0};
                    double dmin = 1e+9;
                    
                    for(double f = (Loss/2)/100; f < Loss/2 - (Loss/2)/200; f+=(Loss/2)/100)
                    {
                        double x0 = getFraction(x[b][d],               f );
                        double x1 = getFraction(x[b][d], 1 - (Loss/2 - f));
                        
                        if(fabs(x1 - x0) < dmin)
                        {
                            limits[0] = x0;
                            limits[1] = x1;
                            
                            dmin = fabs(x1 - x0);
                        }
                    }
                    
                    c[b].lim[d][0] = limits[0];
                    c[b].lim[d][1] = limits[1];
                    //      c[b].lim[d][0] = getFraction(x[b][d],     Loss/2);
                    //      c[b].lim[d][1] = getFraction(x[b][d], 1 - Loss/2);
                }
            
            // Area
            c[b].area = (c[b].lim[0][1] - c[b].lim[0][0]) *
            (c[b].lim[1][1] - c[b].lim[1][0]);
        }
    }
    
    static void printOut(const TH2F* histo, const std::vector<Point>& points, const std::vector<Center>& c,
                         const std::string& flag)
    {
        printToFile(histo, "points_" + flag + ".dat");
        
        std::ofstream limitsFile("limits_" + flag + ".par");
        for(int b=0; b<2; b++) {
            for(int d=0; d<2; d++) {
                limitsFile << " l" << b << d << "=" << c[b].lim[d][0] << std::endl;
                limitsFile << " h" << b << d << "=" << c[b].lim[d][1] << std::endl;
            }
        }
        
        std::ofstream centersFile("centers_" + flag + ".dat");
        for(int b=0; b<2; b++)
            centersFile << " " << c[b].x[0] << " " << c[b].x[1] << std::endl;
    }
    
    void process(const TH2F * histo, std::vector<Center>& c, const std::string& flag, bool print)
    {
        std::vector<Point> points;
        
        for(int i = 1 ; i <= histo->GetNbinsX(); i++)
            for(int j = 1 ; j <= histo->GetNbinsY(); j++)
                if(histo->GetBinContent(i,j) > 0)
                {
                    Point p;
                    p.x[0] = histo->GetXaxis()->GetBinCenter(i);
                    p.x[1] = histo->GetYaxis()->GetBinCenter(j);
                    p.w    = int(histo->GetBinContent(i,j) + 0.5);
                    
                    points.push_back(p);
                }
        
        line = histo->ProjectionY("_py",0,0);
        line->Reset();
        
        kMeans(points, c);
        findLimits(points, c);
        if(print)
            printOut(histo, points,c, flag);
    }
    
    static void printToFile(const TH2F *h2, const std::string& fileName)
    {
        std::ofstream file(fileName);
        
        for(int i = 1; i <= h2->GetNbinsX(); i++) {
            for(int j = 1; j <= h2->GetNbinsY(); j++)
                file << " " << h2->GetXaxis()->GetBinLowEdge(i)
                << " " << h2->GetYaxis()->GetBinLowEdge(j)
                << " " << h2->GetBinContent(i,j)
                << std::endl;
            
            file << " " << h2->GetXaxis()->GetBinLowEdge(i)
            << " " << h2->GetYaxis()->GetXmax()
            << " 0" << std::endl;
            file << std::endl;
        }
        
        for(int j = 1; j <= h2->GetNbinsY(); j++) {
            file << " " << h2->GetXaxis()->GetXmax()
                 << " " << h2->GetYaxis()->GetBinLowEdge(j)
                 << " 0" << std::endl;
        }
        
        file << " " << h2->GetXaxis()->GetXmax()
             << " " << h2->GetYaxis()->GetXmax()
             << " 0" << std::endl;
    }

private:
    TH1D* line{nullptr};
};

int main(int argc, char* argv[])
{
    if(argc != 3) {
        std::cerr << "Usage: pixelAnalyzer <input_file> <output_file>\n"
                  << "Example: pixelAnalyzer clusterShape.root pixelShape.par\n";
        return 1;
    }
    const std::string inputFileName(argv[1]);
    const std::string outputFileName(argv[2]);
    ClusterAnalyzer theClusterAnalyzer;  
    theClusterAnalyzer.analyze(inputFileName, outputFileName);
    
    return 0;
}

