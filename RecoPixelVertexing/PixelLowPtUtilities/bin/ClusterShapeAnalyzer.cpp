#include <cmath>
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <boost/program_options.hpp>

#include "TFile.h"
#include "TH2F.h"

namespace cluster_shape_filter {

  static constexpr size_t n_dim = 2;

  using Position = std::array<double, n_dim>;

  struct Limits {
    double down, up;
    Limits() : down(-std::numeric_limits<double>::infinity()), up(std::numeric_limits<double>::infinity()) {}
    Limits(double _down, double _up) : down(_down), up(_up) {}
    double size() const { return std::abs(down - up); }
  };

  using BoxLimits = std::array<Limits, n_dim>;

  struct Point {
    Position x;
    int weight;
    int cluster_index;
  };

  struct Center {
    Position x;
    int n_clusters;
    BoxLimits limits;

    double distance2(const Point& p) const {
      double d = 0;
      for (size_t n = 0; n < x.size(); ++n)
        d += std::pow(x[n] - p.x[n], 2);
      return d;
    }

    double area() const {
      double s = 1;
      for (size_t n = 0; n < limits.size(); ++n)
        s *= limits[n].size();
      return s;
    }
  };

  using AsymmetricCut = std::array<Center, 2>;

  std::ostream& operator<<(std::ostream& os, const AsymmetricCut& c) {
    for (size_t b = 0; b < c.size(); ++b) {
      for (size_t d = 0; d < c[b].limits.size(); ++d) {
        os << " " << c[b].limits[d].down << " " << c[b].limits[d].up;
      }
    }

    for (size_t b = 0; b < c.size(); ++b) {
      const double v = c[b].area() > 0 ? c[b].n_clusters / c[b].area() : 0.;
      os << " " << v << " " << c[b].n_clusters;
    }
    return os;
  }

  template <typename Object>
  Object* ReadObject(TFile& file, const std::string& name) {
    auto object = dynamic_cast<Object*>(file.Get(name.c_str()));
    if (!object) {
      std::ostringstream ss;
      ss << "Object '" << name << "' with type '" << typeid(Object).name() << "' not found in '" << file.GetName()
         << "'.";
      throw std::runtime_error(ss.str());
    }
    return object;
  }

  struct Arguments {
    std::string input, output, map_output;
    double loss{0.005};
    size_t minEntries{100};
    int exMax{10}, eyMax{15};
    bool verbose{false};
  };

  class ClusterAnalyzer {
  public:
    ClusterAnalyzer(const Arguments& _args) : args(_args) {}

    void analyze() {
      static const std::vector<std::string> subName = {"barrel", "endcap"};

      TFile resFile(args.input.c_str(), "READ");
      if (resFile.IsZombie())
        throw std::runtime_error("Input file not opened.");

      std::ofstream outFile;
      outFile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      outFile.open(args.output);
      std::shared_ptr<std::ofstream> mapFile;
      if (!args.map_output.empty()) {
        mapFile = std::make_shared<std::ofstream>();
        mapFile->exceptions(std::ofstream::failbit | std::ofstream::badbit);
        mapFile->open(args.map_output);
      }

      for (size_t is = 0; is < subName.size(); ++is) {
        for (int ix = 0; ix <= args.exMax; ++ix) {
          std::cout << " " << subName.at(is) << " dx= " << ix << " ";
          for (int iy = 0; iy <= args.eyMax; ++iy) {
            std::ostringstream histName;
            //                    histName << "hrpc_" << is << "_" << ix << "_" << iy;
            histName << "hspc_" << is << "_" << ix << "_" << iy;
            auto histo = ReadObject<TH2F>(resFile, histName.str());

            std::cout << ".";

            // Initial guess
            AsymmetricCut c;
            c[0].n_clusters = 0;
            c[0].x[0] = ix + 1;
            c[0].x[1] = iy + 1;

            c[1].n_clusters = 0;
            c[1].x[0] = -ix - 1;
            c[1].x[1] = -iy - 1;

            std::ostringstream flag;
            flag << subName.at(is) << "_" << ix << "_" << iy;
            process(*histo, c, flag.str());

            if (histo->Integral() >= args.minEntries) {
              // Fix barrel_0_0
              if (is == 0 && ix == 0 && iy == 0) {
                c[0].limits[1].down = 0.;
                c[1].limits[1].up = 0.;
              }

              outFile << is << " " << ix << " " << iy << c << std::endl;
            }

            if (mapFile)
              *mapFile << ix << " " << iy << " " << histo->Integral() << std::endl;
          }
          outFile << std::endl;
          std::cout << std::endl;
          if (mapFile)
            *mapFile << ix << " " << args.eyMax + 1 << " " << 0 << "\n" << std::endl;
        }

        if (mapFile) {
          for (int iy = 0; iy <= args.eyMax + 1; ++iy)
            *mapFile << args.exMax + 1 << " " << iy << " " << 0 << std::endl;
          *mapFile << std::endl << std::endl;
        }

        outFile << std::endl;
      }
    }

  private:
    using HistArray = std::array<std::array<TH1D*, 2>, 2>;

    HistArray CreateHistArray(const std::vector<Point>& points, const TH1D& line) const {
      HistArray x;
      for (size_t n = 0; n < x.size(); ++n) {
        for (size_t d = 0; d < x[n].size(); ++d) {
          std::ostringstream name;
          name << n << "_" << d;
          x[n][d] = dynamic_cast<TH1D*>(line.Clone(name.str().c_str()));
        }
      }

      for (const auto& point : points) {
        for (size_t d = 0; d < x[point.cluster_index].size(); ++d) {
          x[point.cluster_index][d]->Fill(point.x[d], point.weight);
        }
      }
      return x;
    }

    static double getFraction(const TH1D& line, double fraction) {
      double n = line.Integral();
      double s = 0;

      for (int i = 0; i <= line.GetNbinsX() + 1; ++i) {
        s += line.GetBinContent(i);
        if (s > n * fraction) {
          return line.GetXaxis()->GetBinUpEdge(i) -
                 (s - n * fraction) / line.GetBinContent(i) * line.GetXaxis()->GetBinWidth(i);
        }
      }

      throw std::runtime_error("Unable to determine a point for the given fraction.");
    }

    void kMeans(std::vector<Point>& points, AsymmetricCut& c, const TH1D& line) const {
      int changes;
      do {
        changes = 0;

        // Sort point into clusters
        for (auto& point : points) {
          const int n = c[0].distance2(point) < c[1].distance2(point) ? 0 : 1;
          if (n != point.cluster_index) {
            point.cluster_index = n;
            ++changes;
          }
        }

        if (changes != 0) {
          // Re-compute centers
          auto x = CreateHistArray(points, line);

          for (size_t n = 0; n < c.size(); ++n) {
            for (size_t d = 0; d < c[n].x.size(); ++d) {
              c[n].n_clusters = std::ceil(x[n][d]->Integral());
              if (x[n][d]->Integral() > 0)
                c[n].x[d] = getFraction(*x[n][d], 0.5);
            }
          }
        }
      } while (changes != 0);
    }

    void findLimits(const std::vector<Point>& points, AsymmetricCut& c, const TH1D& line) const {
      auto x = CreateHistArray(points, line);

      for (size_t b = 0; b < x.size(); ++b) {
        for (size_t d = 0; d < x[b].size(); ++d) {
          if (x[b][d]->Integral() <= 0)
            continue;
          Limits best_limits;
          for (double f = (args.loss / 2) / 100; f < args.loss / 2 - (args.loss / 2) / 200;
               f += (args.loss / 2) / 100) {
            const Limits limits(getFraction(*x[b][d], f), getFraction(*x[b][d], 1 - (args.loss / 2 - f)));
            if (limits.size() < best_limits.size())
              best_limits = limits;
          }
          c[b].limits[d] = best_limits;
        }
      }
    }

    static void printOut(const TH2F& histo, const AsymmetricCut& c, const std::string& flag) {
      printToFile(histo, "points_" + flag + ".dat");

      std::ofstream limitsFile("limits_" + flag + ".par");
      for (size_t b = 0; b < c.size(); ++b) {
        for (size_t d = 0; d < c[b].limits.size(); ++d) {
          limitsFile << " l" << b << d << "=" << c[b].limits[d].down << std::endl;
          limitsFile << " h" << b << d << "=" << c[b].limits[d].up << std::endl;
        }
      }

      std::ofstream centersFile("centers_" + flag + ".dat");
      for (size_t b = 0; b < c.size(); ++b)
        centersFile << " " << c[b].x[0] << " " << c[b].x[1] << std::endl;
    }

    void process(const TH2F& histo, AsymmetricCut& c, const std::string& flag) {
      std::vector<Point> points;
      points.reserve(histo.GetNbinsX() * histo.GetNbinsY());

      for (int i = 1; i <= histo.GetNbinsX(); ++i) {
        for (int j = 1; j <= histo.GetNbinsY(); ++j) {
          if (histo.GetBinContent(i, j) <= 0)
            continue;
          Point p;
          p.x[0] = histo.GetXaxis()->GetBinCenter(i);
          p.x[1] = histo.GetYaxis()->GetBinCenter(j);
          p.weight = std::ceil(histo.GetBinContent(i, j));
          points.push_back(p);
        }
      }

      auto line = histo.ProjectionY("_py", 0, 0);
      line->Reset();

      kMeans(points, c, *line);
      findLimits(points, c, *line);
      if (args.verbose)
        printOut(histo, c, flag);
    }

    static void printToFile(const TH2F& h2, const std::string& fileName) {
      std::ofstream file;
      file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      file.open(fileName);

      for (int i = 1; i <= h2.GetNbinsX(); i++) {
        for (int j = 1; j <= h2.GetNbinsY(); j++)
          file << " " << h2.GetXaxis()->GetBinLowEdge(i) << " " << h2.GetYaxis()->GetBinLowEdge(j) << " "
               << h2.GetBinContent(i, j) << "\n";

        file << " " << h2.GetXaxis()->GetBinLowEdge(i) << " " << h2.GetYaxis()->GetXmax() << " 0\n\n";
      }

      for (int j = 1; j <= h2.GetNbinsY(); j++) {
        file << " " << h2.GetXaxis()->GetXmax() << " " << h2.GetYaxis()->GetBinLowEdge(j) << " 0\n";
      }

      file << " " << h2.GetXaxis()->GetXmax() << " " << h2.GetYaxis()->GetXmax() << " 0" << std::endl;
    }

  private:
    const Arguments args;
  };

}  // namespace cluster_shape_filter

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  using namespace cluster_shape_filter;

  Arguments args;
  po::options_description desc("Cluster shape filter analyzer");
  std::ostringstream ss_loss;
  ss_loss << std::setprecision(4) << args.loss;
  // clang-format off
  desc.add_options()
      ("help", "print help message")
      ("input", po::value<std::string>(&args.input)->required(), "input file with extracted cluster shapes")
      ("output", po::value<std::string>(&args.output)->required(), "output calibration file")
      ("map-output", po::value<std::string>(&args.map_output)->default_value(""), "output calibration file")
      ("loss",
       po::value<double>(&args.loss)->default_value(args.loss, ss_loss.str()),
       "(1 - efficiency) threshold for the WP definition")
      ("min-entries", po::value<size_t>(&args.minEntries)->default_value(args.minEntries), "")
      ("ex-max", po::value<int>(&args.exMax)->default_value(args.exMax), "")
      ("ey-max", po::value<int>(&args.eyMax)->default_value(args.eyMax), "")
      ("verbose", po::bool_switch(&args.verbose));
  // clang-format on

  try {
    po::variables_map variables;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), variables);
    if (variables.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    po::notify(variables);

    cluster_shape_filter::ClusterAnalyzer analyzer(args);
    analyzer.analyze();

  } catch (po::error& e) {
    std::cerr << "ERROR: " << e.what() << ".\n\n" << desc << std::endl;
  } catch (std::exception& e) {
    std::cerr << "\nERROR: " << e.what() << std::endl;
  }

  return 0;
}
