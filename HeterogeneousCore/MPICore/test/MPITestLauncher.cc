// MPI Test Launcher - main entry point
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "MPITestBase.h"
#include "MPITestMaster.h"
#include "MPITestWorker.h"

// Available MPI communication methods (name -> {id, description})
struct MethodInfo {
  int id;
  const char* description;
};
const std::map<std::string, MethodInfo> MPI_METHODS = {
    {"blockingSend", {4, "Point-to-point using MPI_Send/MPI_Recv"}},
    {"nonBlockingSend", {3, "Point-to-point using MPI_Issend/MPI_Irecv"}},
    {"blockingScatter", {2, "Collective using MPI_Scatter/MPI_Gather"}},
    {"nonBlockingScatter", {1, "Collective using MPI_Iscatter/MPI_Igather"}},
    {"blockingSendRecv", {6, "Combined send/receive using MPI_Sendrecv"}},
    {"oneSidedCommMaster", {5, "One-sided RMA (master-initiated put/get)"}},
    {"oneSidedCommWorker", {7, "One-sided RMA (worker-initiated put/get)"}}};

void printHelp(const char* programName) {
  std::cout << "\nUsage: mpirun -np <nprocs> " << programName << " [options]\n\n"
            << "Options:\n"
            << "  -s, --size <n>        Vector size for the test (default: 100)\n"
            << "  -i, --iterations <n>  Number of iterations to average (default: 10)\n"
            << "  -w, --warmup <n>      Number of warmup iterations to skip (default: 0)\n"
            << "  -t, --tests <list>    Comma-separated list of tests to run, or 'all' (default: all)\n"
            << "  -o, --output-csv      Write results to CSV files (default: off)\n"
            << "  -h, --help            Show this help message\n\n"
            << "Available tests:\n"
            << "  all                      Run all tests\n";
  for (const auto& [name, info] : MPI_METHODS) {
    std::cout << "  " << std::left << std::setw(22) << name << " " << info.description << "\n";
  }
  std::cout << "\nExample:\n"
            << "  mpirun -np 4 " << programName << " -s 1000 -i 20 -t blockingSend,nonBlockingSend\n"
            << "  mpirun -np 4 " << programName << " -t all\n"
            << std::endl;
}

std::tuple<int, std::vector<int>, int, int, int, bool> parseCommands(int argc, char* argv[]) {
  // Default values
  int vecSize = 100;
  int iterations = 10;
  int warmup = 0;
  std::vector<int> commMethods;
  int inputNum_ = 0;
  bool outputCsv = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      printHelp(argv[0]);
      exit(0);
    } else if (arg == "-s" || arg == "--size") {
      if (++i >= argc) {
        std::cerr << "Error: --size requires an argument\n";
        abort();
      }
      try {
        vecSize = std::stoi(argv[i]);
      } catch (std::exception& err) {
        std::cerr << "Error: --size must be an integer: " << err.what() << std::endl;
        abort();
      }
    } else if (arg == "-i" || arg == "--iterations") {
      if (++i >= argc) {
        std::cerr << "Error: --iterations requires an argument\n";
        abort();
      }
      try {
        iterations = std::stoi(argv[i]);
      } catch (std::exception& err) {
        std::cerr << "Error: --iterations must be an integer: " << err.what() << std::endl;
        abort();
      }
    } else if (arg == "-w" || arg == "--warmup") {
      if (++i >= argc) {
        std::cerr << "Error: --warmup requires an argument\n";
        abort();
      }
      try {
        warmup = std::stoi(argv[i]);
      } catch (std::exception& err) {
        std::cerr << "Error: --warmup must be an integer: " << err.what() << std::endl;
        abort();
      }
    } else if (arg == "-t" || arg == "--tests") {
      if (++i >= argc) {
        std::cerr << "Error: --tests requires an argument\n";
        abort();
      }
      std::string testList = argv[i];
      if (testList == "all") {
        for (const auto& [name, info] : MPI_METHODS) {
          commMethods.push_back(info.id);
        }
      } else {
        std::stringstream ss(testList);
        std::string testName;
        while (std::getline(ss, testName, ',')) {
          auto it = MPI_METHODS.find(testName);
          if (it == MPI_METHODS.end()) {
            std::cerr << "Error: Unknown test '" << testName << "'\n";
            std::cerr << "Use --help to see available tests\n";
            abort();
          }
          commMethods.push_back(it->second.id);
          inputNum_ = inputNum_ * 10 + it->second.id;  // Build numeric representation for CSV naming
        }
      }
    } else if (arg == "-o" || arg == "--output-csv") {
      outputCsv = true;
    } else {
      std::cerr << "Error: Unknown option '" << arg << "'\n";
      std::cerr << "Use --help for usage information\n";
      abort();
    }
  }

  // If no tests specified, run all tests
  if (commMethods.empty()) {
    for (const auto& [name, info] : MPI_METHODS) {
      commMethods.push_back(info.id);
    }
  }

  return std::make_tuple(vecSize, commMethods, iterations, warmup, inputNum_, outputCsv);
}

// main function
int main(int argc, char* argv[]) {
  auto [vecSize, commMethods, iterations, warmup, inputNum_, outputCsv] = parseCommands(argc, argv);

  MPI_Init(&argc, &argv);

  int rank;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size < 2) {
    if (rank == 0) {
      std::cerr << "Error: This program requires at least 2 MPI processes.\n";
      std::cerr << "Run with: mpirun -np <nprocs> " << argv[0] << " [options]\n";
      std::cerr << "Use --help for more information.\n";
    }
    MPI_Finalize();
    return 1;
  }

  std::unique_ptr<MPI_TEST> MPIObject;

  if (rank == 0) {
    MPIObject = std::make_unique<MPI_Master>(vecSize);
  } else {
    MPIObject = std::make_unique<MPI_Worker>();
  }

  MPIObject->getTimeMeasurements(vecSize, commMethods, iterations, warmup, inputNum_, outputCsv);

  MPI_Finalize();

  return 0;
}
