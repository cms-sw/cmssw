#ifndef PROCESSINTERFACE_H
#define PROCESSINTERFACE_H

#include <vector>
#include <unistd.h>
#include <fcntl.h>
class MPI_TEST {
public:
  int size_;
  int rank_;
  std::vector<float> v1_;
  std::vector<float> v2_;
  std::vector<float> result_;

  const int precisionFactor = 4;

  MPI_TEST() {
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  }

  virtual std::pair<float, float> blockingSend() = 0;
  virtual std::pair<float, float> nonBlockingSend() = 0;
  virtual std::pair<float, float> blockingScatter() = 0;
  virtual std::pair<float, float> nonBlockingScatter() = 0;
  virtual std::pair<float, float> blockingSendRecv() = 0;
  virtual std::pair<float, float> oneSidedCommMaster() = 0;
  virtual std::pair<float, float> oneSidedCommWorker() = 0;

  void getTimeMeasurements(int vecSize, std::vector<int> commMethods, int iterations, int inputNum_) {
    //tuple<commMethod, avgSendTime, avgRecvTime>

    std::vector<std::tuple<int, float, float>> results;

    for (auto i = 0; i < commMethods.size(); ++i) {
      auto [avgSendTime, avgRecvTime] = calculateAverageTime(commMethods[i], iterations);
      results.push_back(std::make_tuple(commMethods[i], avgSendTime, avgRecvTime));
    }

    if (rank_ == 0) {                                            //root
      printResults(results, iterations, vecSize, size_);         //print to std output
      printCSV(results, iterations, vecSize, size_, inputNum_);  //print to a csv file
    }
  }
  std::pair<float, float> calculateAverageTime(int funcNum, int iterations) {
    std::pair<float, float> averageTime;

    for (unsigned int i = 0; i < iterations; ++i) {
      std::pair<float, float> timeDuration;
      switch (funcNum) {
        case 1:
          timeDuration = nonBlockingScatter();
          break;
        case 2:
          timeDuration = blockingScatter();
          break;
        case 3:
          timeDuration = nonBlockingSend();
          break;
        case 4:
          timeDuration = blockingSend();
          break;
        case 5:
          timeDuration = oneSidedCommMaster();
          break;
        case 6:
          timeDuration = blockingSendRecv();
          break;
        case 7:
          timeDuration = oneSidedCommWorker();
          break;
        default:
          std::cerr << "\n\n\tError: Invalid function number!\n";
          abort();
      }

      averageTime.first += timeDuration.first;    //sendDuration from root
      averageTime.second += timeDuration.second;  //recvDuration to root
    }

    // Calculate the average timings by dividing the accumulated values
    averageTime.first /= iterations;
    averageTime.second /= iterations;

    return averageTime;
  }

  void generateRandomData(int vectorSize) {
    std::random_device rand;                       // Random device used to seed the random engine.
    std::default_random_engine gener(rand());      // Default random engine.
    std::uniform_real_distribution<> dis(0., 1.);  // Uniform distribution from 0 to 1.
    // Generate a random number and assign it to the vector element.
    for (int i = 0; i < vectorSize; i++) {
      v1_.push_back(dis(gener));
      v2_.push_back(dis(gener));
      //validationReference_.push_back(mainInput1_[i] + mainInput2_[i]);
    }
  }

  void checkResult(std::vector<float> result_Vect) {
    float totalError{0.0};  // Variable to store the total error.

    // Calculate the percentage difference and accumulate the total error.
    for (int i = 0; i < result_Vect.size(); i++) {
      float s = v1_[i] + v2_[i];
      totalError += ((s - result_Vect[i]) / s) * 100.0;
    }

    // If there is a non-zero total error, print the result_s and error analysis.
    if (totalError == 0.0) {
      return;  // No error Found;
    }

    std::cout << "\n-------------------------------------------------------\n";
    std::cout << "| RootSum | WorksSum | Error   | Error %  | Process # |";
    std::cout << "\n-------------------------------------------------------\n";
    std::cout.precision(precisionFactor);

    int batchSize = v1_.size() / (size_ - 1);     //execluding root
    int extraBatches = v1_.size() % (size_ - 1);  //execluding root
    int curBatchSize = batchSize + (extraBatches > 0) ? 1 : 0;
    int workerRank = 0;
    for (int i = 0; i < result_Vect.size(); i++) {
      float correct = v1_[i] + v2_[i];
      float error = correct - result_Vect[i];
      if (error != 0.0) {
        float errorPercent = (error / correct) * 100.0;
        std::cout << "| " << correct << "  | " << result_Vect[i] << "  |" << std::setw(9) << error << " |"
                  << std::setw(9) << errorPercent << " |" << std::setw(9) << workerRank << " |\n";
      }
      if (i > curBatchSize) {
        workerRank += 1;
        curBatchSize = batchSize + (extraBatches - workerRank > 0) ? 1 : 0;
      }
    }

    std::cout << "-------------------------------------------------------\n";
    std::cout << "Total Error = " << totalError << std::endl;
  }

  // print to the standared output
  void printResults(const std::vector<std::tuple<int, float, float>> executionTimes,
                    int iterations,
                    int vecSize,
                    int size) {
    /*Printing to the output screen*/
    const std::string COMM_METHOD_NAMES[] = {"NONBLOCKING SCATTER",
                                             "BLOCKING SCATTER",
                                             "BLOCKING SEND/RECV",
                                             "NONBLOCKING SEND/RECV",
                                             "ONE SIDED MASTER",
                                             "BLOCKING SENDRECV",
                                             "ONE SIDED WORKER"};
    const auto COL1 = 25, COL2 = 15, COL3 = 15, COL4 = 11;
    std::string ROW =
        "============================================================================================================";
    std::string DASHES =
        "------------------------------------------------------------------------------------------------------------";
    std::cout.flags(std::ios::fixed | std::ios::showpoint);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(4);

    std::cout << "\n\n\t" << ROW;
    std::cout << "\n\t|| " << std::left << std::setw(COL1) << "Communication Method"
              << "|| " << std::setw(COL2) << "Send"
              << "|| " << std::setw(COL3) << "Receive"
              << "|| " << std::setw(COL4) << "Iterations"
              << "|| " << std::setw(COL4) << "Vector Size"
              << "|| " << std::setw(COL4) << "Processes"
              << "||"
              << "\n\t" << ROW;

    // Print the execution times and related information
    for (int i = 0; i < executionTimes.size(); ++i) {
      if (i > 0)
        std::cout << "\n\t" << DASHES;

      auto [commMethod, avgSendTime, avgRecvTime] = executionTimes[i];

      std::cout << "\n\t|| " << std::left << std::setw(COL1) << COMM_METHOD_NAMES[commMethod - 1] << "|| "
                << std::setw(COL2) << avgSendTime << "|| " << std::setw(COL3) << avgRecvTime << "|| " << std::setw(COL4)
                << iterations << "|| " << std::setw(COL4) << vecSize << "|| " << std::setw(COL4) << size << "||";
    }

    std::cout << "\n\t" << ROW << "\n\n";
  }  //end of printResults

  // printing to a text file
  void printFile(const std::vector<std::tuple<int, float, float>> executionTimes,
                 int iterations,
                 int vecSize,
                 int size) {
    /*Printing to the file*/
    /*Printing to the output screen*/
    const std::string COMM_METHOD_NAMES[] = {"NONBLOCKING SCATTER",
                                             "BLOCKING SCATTER",
                                             "BLOCKING SEND/RECV",
                                             "NONBLOCKING SEND/RECV",
                                             "ONE SIDED MASTER",
                                             "BLOCKING SENDRECV",
                                             "ONE SIDED WORKER"};
    const auto COL1 = 25, COL2 = 15, COL3 = 15, COL4 = 11;
    std::string ROW =
        "============================================================================================================";
    std::string DASHES =
        "------------------------------------------------------------------------------------------------------------";
    std::cout.flags(std::ios::fixed | std::ios::showpoint);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(4);

    std::fstream fd;
    fd.open("result_file.txt", std::fstream::in | std::fstream::out | std::fstream::app);
    if (!fd) {  //file cannot be opened
      std::cout << "File Cannot be opened1\n";
      exit(0);
    } else {  //file is opened

      //write to the file the results

      fd << "\n\n\t" << ROW;
      fd << "\n\t|| " << std::left << std::setw(COL1) << "Communication Method"
         << "|| " << std::setw(COL2) << "Scatter/Send"
         << "|| " << std::setw(COL3) << "Gather/Receive"
         << "|| " << std::setw(COL4) << "Iterations"
         << "|| " << std::setw(COL4) << "Vector Size"
         << "|| " << std::setw(COL4) << "Processes"
         << "||"
         << "\n\t" << ROW;

      // Print the execution times and related information
      for (int i = 0; i < executionTimes.size(); ++i) {
        if (i > 0)
          fd << "\n\t" << DASHES;

        auto [commMethod, avgSendTime, avgRecvTime] = executionTimes[i];

        fd << "\n\t|| " << std::left << std::setw(COL1) << COMM_METHOD_NAMES[commMethod - 1] << "|| " << std::setw(COL2)
           << avgSendTime << "|| " << std::setw(COL3) << avgRecvTime << "|| " << std::setw(COL4) << iterations << "|| "
           << std::setw(COL4) << vecSize << "|| " << std::setw(COL4) << size << "||";
      }

      fd << "\n\t" << ROW << "\n\n";
      fd.close();
    }

  }  //end of printFile
  // printing to a csv file
  void printCSV(const std::vector<std::tuple<int, float, float>> executionTimes,
                int iterations,
                int vecSize,
                int size,
                int inputNum) {
    /*Printing to the file*/
    const std::string COMM_METHOD_NAMES[] = {"NONBLOCKING SCATTER",
                                             "BLOCKING SCATTER",
                                             "BLOCKING SEND/RECV",
                                             "NONBLOCKING SEND/RECV",
                                             "ONE SIDED MASTER",
                                             "BLOCKING SENDRECV",
                                             "ONE SIDED WORKER"};
    std::fstream fd;
    int c = 0;
    while (inputNum > 0) {
      int digit = inputNum % 10;
      inputNum /= 10;
      auto [commMethod, avgSendTime, avgRecvTime] = executionTimes[c];
      c += 1;
      if (COMM_METHOD_NAMES[digit - 1] == "NONBLOCKING SCATTER") {
        fd.open("NB_scatter.csv", std::fstream::in | std::fstream::out | std::fstream::app);

      } else if (COMM_METHOD_NAMES[digit - 1] == "BLOCKING SCATTER") {
        fd.open("B_scatter.csv", std::fstream::in | std::fstream::out | std::fstream::app);
      } else if (COMM_METHOD_NAMES[digit - 1] == "BLOCKING SEND/RECV") {
        fd.open("B_send_Recv.csv", std::fstream::in | std::fstream::out | std::fstream::app);
      } else if (COMM_METHOD_NAMES[digit - 1] == "NONBLOCKING SEND/RECV") {
        fd.open("NB_send_Recv.csv", std::fstream::in | std::fstream::out | std::fstream::app);
      } else if (COMM_METHOD_NAMES[digit - 1] == "BLOCKING SENDRECV") {
        fd.open("B_sendRecv.csv", std::fstream::in | std::fstream::out | std::fstream::app);

      } else if (COMM_METHOD_NAMES[digit - 1] == "ONE SIDED MASTER") {
        fd.open("NB_oneSidedM.csv", std::fstream::in | std::fstream::out | std::fstream::app);

      } else if (COMM_METHOD_NAMES[digit - 1] == "ONE SIDED WORKER") {
        fd.open("NB_oneSidedW.csv", std::fstream::in | std::fstream::out | std::fstream::app);
      } else {
        std::cout << "File name not found!\n";
      }
      if (!fd) {  //file cannot be opened
        std::cout << "File Cannot be opened!\n";
        exit(0);
      }

      fd.seekg(0, std::ios::end);  //seek the end of the file
      int file_size = fd.tellg();
      if (file_size == 0)  //first time to open and write to a file
      {
        fd << "Communication Method"
           << ","
           << "Scatter/Send"
           << ","
           << "Gather/Receive"
           << ","
           << "Iterations"
           << ","
           << "Vector Size"
           << ","
           << "Processes"
           << "\n";
        // Print the execution times and related information
        fd << COMM_METHOD_NAMES[commMethod - 1] << "," << avgSendTime << "," << avgRecvTime << "," << iterations << ","
           << vecSize << "," << size;

      } else {  //not the first time i.e. data exist in the file
        fd << COMM_METHOD_NAMES[commMethod - 1] << "," << avgSendTime << "," << avgRecvTime << "," << iterations << ","
           << vecSize << "," << size;
      }
      fd << "\n";
      fd.close();
    }  //end while
  }    //end of printCSV
};

#endif  // PROCESSINTERFACE_H
