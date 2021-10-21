#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
#include <mpi.h>
#include <unistd.h>

// std::cout << "\n\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
// std::cout << "\n\t+ (1)  Non Blocking Scatter.                             +";
// std::cout << "\n\t+ (2)  Blocking Scatter.                                 +";
// std::cout << "\n\t+ (3)  Non Blocking Send and Receive.                    +";
// std::cout << "\n\t+ (4)  Blocking Send and Receive.                        +";
// std::cout << "\n\t+ (5)  Non Blocking Send and Receive with Multiple Tasks +";
// std::cout << "\n\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";

struct MPIData {
  int num_procs{0};
  int rank{0};

  std::pair<int, int> workSplit;
  std::vector<float> input1;          //declare vector 1.
  std::vector<float> input2;          //declare vector 2.
  std::vector<float> output;          //declare vector fulled only by root to get result from workers.
  std::vector<float> reference;       //declare vector to verify the ruslt form each process.
  std::vector<float> vectorWorkers1;  //declare vector 1 for workers only.
  std::vector<float> vectorWorkers2;  //declare vector 2 for workers only.
  std::vector<int> displacement;      //declare vector for selecting location of each element to be sent.
  std::vector<int> numberToSend;
};

int choices = 5;             //number of functions of MPI.
int size = 21;               //default size of vectors.
unsigned int runNumber = 5;  //default number of time to run each function in order to take the average.
int precision = 4;           //default digits after decimal point.
int function = 5;            //Total number of functions in the program.
int root = 0;
int choice = 0;  //user Choice to select function to run.

std::vector<int> userChoices(1, 1);  //save convertion integer to vector.

const std::vector<int> chooseFunction(int toInteger);  //Convert integers to a vector.
std::vector<std::pair<float, float>> timing(
    choices, std::make_pair(0, 0));  //to save time of scatter/send and gather/receive for each function.

void randomGenerator(std::vector<float>& vect);                    //generate uniform random numbers.
std::pair<int, int> splitProcess(int works, int numberOfProcess);  //calcualte for each process number of works.
const std::vector<int> numberDataSend(
    int numberOfProcess, std::pair<int, int> splitWorks);  //findout number of data to be sent for each process.
const std::vector<int> displacmentData(
    int numberOfProcess,
    std::pair<int, int> splitWorks,
    const std::vector<int>& numberDataSend);  //findout the index of data to be sent for each process
void checkingResultsPrintout(std::vector<float>& reference,
                             std::vector<float>& output,
                             std::pair<int, int> workSplit,
                             const std::vector<int>& displacement,
                             const std::vector<int>& numberDataSend);

const std::pair<float, float> nonBlockScatter(MPIData& mpiInput);
const std::pair<float, float> blockScatter(MPIData& mpiInput);
const std::pair<float, float> nonBlockSend(MPIData& mpiInput);
const std::pair<float, float> blockSend(MPIData& mpiInput);
const std::pair<float, float> multiNonBlockSend(MPIData& mpiInput);

void compare(const std::vector<std::pair<float, float>>& timing,
             int choices,
             const std::vector<int>& digits,
             int runNumber);  //to printout the time for each function that user chose.

const std::pair<float, float> returnAverage(const std::pair<float, float> (*mpiFunctions)(MPIData&),
                                            MPIData& mpiInput,
                                            unsigned int runNumber);  //to get the average of time for each function.

int main(int argc, char* argv[]) {
  int c;

  while ((c = getopt(argc, argv, "s:r:n:")) != -1) {
    switch (c) {
      case 's':
        try {
          size = std::stoll(optarg, nullptr, 0);
        } catch (std::exception& err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'r':
        try {
          //size = std::stoll(optarg, nullptr, 0);
          choice = std::stoll(optarg, nullptr, 0);
          userChoices = chooseFunction(choice);
          //runNumber = std::stoll(argv[3], nullptr, 0);
        } catch (std::exception& err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      case 'n':
        try {
          //size = std::stoll(argv[1], nullptr, 0);
          //choice = std::stoll(argv[2], nullptr, 0);
          //userChoices = chooseFunction(choice);
          runNumber = std::stoll(optarg, nullptr, 0);
        } catch (std::exception& err) {
          std::cout << "\n\tError Must be integer Argument!";
          std::cout << "\n\t" << err.what() << std::endl;
          return 0;
        }
        break;
      default:
        abort();
    }
  }

  //   if (argc == 2) {
  //     try {
  //       size = std::stoll(argv[1], nullptr, 0);
  //     } catch (std::exception& err) {
  //       std::cout << "\n\tError Must be integer Argument!";
  //       std::cout << "\n\t" << err.what() << std::endl;
  //       return 0;
  //     }
  //   } else if (argc == 3) {
  //     try {
  //       size = std::stoll(argv[1], nullptr, 0);
  //       choice = std::stoll(argv[2], nullptr, 0);
  //       userChoices = chooseFunction(choice);
  //     } catch (std::exception& err) {
  //       std::cout << "\n\tError Must be integer Argument!";
  //       std::cout << "\n\t" << err.what() << std::endl;
  //       return 0;
  //     }
  //   }else if (argc > 3) {
  //     try {
  //       size = std::stoll(argv[1], nullptr, 0);
  //       choice = std::stoll(argv[2], nullptr, 0);
  //       userChoices = chooseFunction(choice);
  //       runNumber = std::stoll(argv[3], nullptr, 0);
  //     } catch (std::exception& err) {
  //       std::cout << "\n\tError Must be integer Argument!";
  //       std::cout << "\n\t" << err.what() << std::endl;
  //       return 0;
  //     }
  //   }

  MPIData mpiInputs;  //greate object from structur to pass into MPI functios.

  MPI_Init(&argc, &argv);                            //initialize communicator environment.
  mpiInputs.num_procs = MPI::COMM_WORLD.Get_size();  //get total size of processes.
  mpiInputs.rank = MPI::COMM_WORLD.Get_rank();       //get each process number.

  mpiInputs.input1.resize(size);  //initialize size.
  mpiInputs.input2.resize(size);
  mpiInputs.output.resize(size);
  mpiInputs.reference.resize(size);

  mpiInputs.workSplit = splitProcess(size, mpiInputs.num_procs);

  if (!mpiInputs.workSplit.first) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return 0;
  }

  mpiInputs.numberToSend = numberDataSend(mpiInputs.num_procs, mpiInputs.workSplit);
  mpiInputs.displacement = displacmentData(mpiInputs.num_procs, mpiInputs.workSplit, mpiInputs.numberToSend);

  mpiInputs.vectorWorkers1.resize(
      mpiInputs.numberToSend[mpiInputs.rank]);  //Resizing each process with appropriate Receiving Data.
  mpiInputs.vectorWorkers2.resize(mpiInputs.numberToSend[mpiInputs.rank]);

  if (!mpiInputs.rank)  //Only for root
  {
    randomGenerator(mpiInputs.input1);  //generate random floating numbers from(0,1) Only in the root.
    randomGenerator(mpiInputs.input2);
    std::cout << "\n\tNumber of Processes " << mpiInputs.num_procs << std::endl;
    std::cout << "\tNumber of workSplit First " << mpiInputs.workSplit.first << std::endl;
    std::cout << "\tNumber of workSplit Second " << mpiInputs.workSplit.second << std::endl;
    for (int j = 0; j < size; j++) {
      mpiInputs.reference[j] = mpiInputs.input1[j] + mpiInputs.input2[j];  //Summing for verification.
    }
  }

  for (long unsigned int i = 0; i < userChoices.size(); ++i) {
    if (userChoices[i] == 1) {
      timing[0] = returnAverage(nonBlockScatter, mpiInputs, runNumber);

    } else if (userChoices[i] == 2) {
      timing[1] = returnAverage(blockScatter, mpiInputs, runNumber);

    } else if (userChoices[i] == 3) {
      timing[2] = returnAverage(nonBlockSend, mpiInputs, runNumber);

    } else if (userChoices[i] == 4) {
      timing[3] = returnAverage(blockSend, mpiInputs, runNumber);

    } else if (userChoices[i] == 5) {
      timing[4] = returnAverage(multiNonBlockSend, mpiInputs, runNumber);

    } else {
      std::cout << "\n\n\tError the User has not chose any number of Function!\n";
      break;
    }
  }

  if (!mpiInputs.rank) {
    compare(timing, choices, userChoices, runNumber);
  }

  MPI::Finalize();

  return 0;
}

void randomGenerator(std::vector<float>& vect) {
  std::random_device rand;
  std::default_random_engine gener(rand());
  std::uniform_real_distribution<> dis(0., 1.);
  int size = vect.size();
  for (int i = 0; i < size; i++) {
    vect.at(i) = dis(gener);
  }
}

std::pair<int, int> splitProcess(int works, int numberOfProcess) {
  std::pair<int, int> Return{0, 0};
  if (numberOfProcess > 1 && numberOfProcess <= works) {
    Return.first = works / (numberOfProcess - 1);   //number of cycle for each process.
    Return.second = works % (numberOfProcess - 1);  //extra cycle for process.
  } else {
    std::cout << "\tError Either No worker are found OR Number Processes Larger than Length!!!\n";
  }

  return Return;
}
const std::vector<int> numberDataSend(int numberOfProcess, std::pair<int, int> splitWorks) {
  std::vector<int> dataSend(numberOfProcess, splitWorks.first);
  dataSend[0] = 0;
  for (int i = 1; i < splitWorks.second + 1; i++)  //neglect root
  {
    dataSend[i] += 1;  //extra work for each first processes.
  }
  return dataSend;
}
const std::vector<int> displacmentData(int numberOfProcess,
                                       std::pair<int, int> splitWorks,
                                       const std::vector<int>& numberDataSend) {
  std::vector<int> displacment(numberOfProcess, splitWorks.first);

  displacment[0] = 0;
  displacment[1] = 0;  //start Here.

  for (int i = 2; i < numberOfProcess; i++)  //neglect root
  {
    displacment[i] = numberDataSend[i - 1] + displacment[i - 1];  //extra work for each first processes.
  }
  return displacment;
}

void checkingResultsPrintout(std::vector<float>& reference,
                             std::vector<float>& output,
                             std::pair<int, int> workSplit,
                             const std::vector<int>& displacement,
                             const std::vector<int>& numberDataSend) {
  float percent{0.0};
  float totalError{0.0};
  int p{1};
  for (int j = 0; j < size; j++) {
    percent = ((reference[j] - output[j]) / reference[j]) * 100;
    totalError += percent;
  }
  if (totalError) {
    std::cout << "\n-------------------------------------------------------\n";
    std::cout << "| RootSum | WorksSum | Error   | Error %  | Process # |";
    std::cout << "\n-------------------------------------------------------\n";
    std::cout.precision(precision);
    for (int j = 0; j < size; j++) {
      std::cout << "| " << reference[j] << "  | " << output[j] << "  |" << std::setw(9) << reference[j] - output[j]
                << " |" << std::setw(9) << percent << " |" << std::setw(9) << p << " |\n";

      if (j + 1 == displacement[p + 1]) {
        ++p;
      }
    }
    std::cout << "-------------------------------------------------------\n";
    std::cout << "-Total Error is " << totalError << std::endl;
    for (long unsigned int j = 1; j < displacement.size(); j++) {
      std::cout << "Process [" << j << "]"
                << " Worked On " << numberDataSend[j] << " Data\n";
    }
  }
}

const std::pair<float, float> nonBlockScatter(MPIData& mpiInput) {
  std::pair<float, float> returnValue;
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  double startTimeScatter = 0;
  double endTimeScatter = 0;
  double startTimeGather = 0;
  double endTimeGather = 0;

  MPI_Request requestRootScatter[2];
  MPI_Request requestRootGather;

  startTimeScatter = MPI_Wtime();  //get time before scattering.

  //Non-Blocking Scatter.
  MPI_Iscatterv(&mpiInput.input1[0],
                &mpiInput.numberToSend[0],
                &mpiInput.displacement[0],
                MPI_FLOAT,
                &mpiInput.vectorWorkers1[0],
                mpiInput.numberToSend[mpiInput.rank],
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD,
                &requestRootScatter[0]);
  MPI_Iscatterv(&mpiInput.input2[0],
                &mpiInput.numberToSend[0],
                &mpiInput.displacement[0],
                MPI_FLOAT,
                &mpiInput.vectorWorkers2[0],
                mpiInput.numberToSend[mpiInput.rank],
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD,
                &requestRootScatter[1]);
  MPI_Waitall(2, requestRootScatter, MPI_STATUS_IGNORE);

  endTimeScatter = MPI_Wtime();  //get time after scattering.

  if (mpiInput.rank)  //Only for Workers
  {
    for (long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++) {
      mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
    }
  }

  startTimeGather = MPI_Wtime();  //get time before Gathering.

  //Non Blocking Gathering.
  MPI_Igatherv(&mpiInput.vectorWorkers1[0],
               mpiInput.numberToSend[mpiInput.rank],
               MPI_FLOAT,
               &mpiInput.output[0],
               &mpiInput.numberToSend[0],
               &mpiInput.displacement[0],
               MPI_FLOAT,
               0,
               MPI_COMM_WORLD,
               &requestRootGather);

  MPI_Wait(&requestRootGather, MPI_STATUS_IGNORE);
  endTimeGather = MPI_Wtime();  //get time after Gathering.

  if (!mpiInput.rank)  //Only root print out the results.
  {
    checkingResultsPrintout(
        mpiInput.reference, mpiInput.output, mpiInput.workSplit, mpiInput.displacement, mpiInput.numberToSend);
    returnValue.first = (endTimeScatter - startTimeScatter) * 1000;
    returnValue.second = (endTimeGather - startTimeGather) * 1000;
  }
  return returnValue;
}
const std::pair<float, float> blockScatter(MPIData& mpiInput) {
  std::pair<float, float> returnValue;
  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  double startTimeScatter = 0;
  double endTimeScatter = 0;
  double startTimeGather = 0;
  double endTimeGather = 0;

  //Blocking Scattering.
  mpiInput.vectorWorkers1.resize(
      mpiInput.numberToSend[mpiInput.rank]);  //Resizing each process with appropriate Receiving Data.
  mpiInput.vectorWorkers2.resize(mpiInput.numberToSend[mpiInput.rank]);

  startTimeScatter = MPI_Wtime();
  MPI_Scatterv(&mpiInput.input1[0],
               &mpiInput.numberToSend[0],
               &mpiInput.displacement[0],
               MPI_FLOAT,
               &mpiInput.vectorWorkers1[0],
               mpiInput.numberToSend[mpiInput.rank],
               MPI_FLOAT,
               0,
               MPI_COMM_WORLD);
  MPI_Scatterv(&mpiInput.input2[0],
               &mpiInput.numberToSend[0],
               &mpiInput.displacement[0],
               MPI_FLOAT,
               &mpiInput.vectorWorkers2[0],
               mpiInput.numberToSend[mpiInput.rank],
               MPI_FLOAT,
               0,
               MPI_COMM_WORLD);
  endTimeScatter = MPI_Wtime();

  if (mpiInput.rank)  //Only for Workers
  {
    for (long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++) {
      mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
    }
  }

  startTimeGather = MPI_Wtime();
  //Blocking Gathering.
  MPI_Gatherv(&mpiInput.vectorWorkers1[0],
              mpiInput.numberToSend[mpiInput.rank],
              MPI_FLOAT,
              &mpiInput.output[0],
              &mpiInput.numberToSend[0],
              &mpiInput.displacement[0],
              MPI_FLOAT,
              0,
              MPI_COMM_WORLD);

  endTimeGather = MPI_Wtime();

  if (!mpiInput.rank)  //Only root print out the results.
  {
    checkingResultsPrintout(
        mpiInput.reference, mpiInput.output, mpiInput.workSplit, mpiInput.displacement, mpiInput.numberToSend);
    returnValue.first = (endTimeScatter - startTimeScatter) * 1000;
    returnValue.second = (endTimeGather - startTimeGather) * 1000;
  }
  return returnValue;
}
const std::pair<float, float> nonBlockSend(MPIData& mpiInput) {
  std::pair<float, float> returnValue;
  double startTimeRootSend = 0;
  double endTimeRootSend = 0;
  double startTimeRootRecv = 0;
  double endTimeRootRecv = 0;

  MPI_Request requestRootSend[2];
  MPI_Request requestRootRecv;
  MPI_Request requestWorkerSend;
  MPI_Request requestWorkerRecv[1];

  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  if (!mpiInput.rank)  //Only for root
  {
    // std::cout << "\n\t\tNon-Blocking Send and Receive " << std::endl;
    startTimeRootSend = MPI_Wtime();
    for (int i = 1; i < mpiInput.num_procs; i++) {
      MPI_Issend(&mpiInput.input1[mpiInput.displacement[i]],
                 mpiInput.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 &requestRootSend[0]);  //Tag is 0
      MPI_Issend(&mpiInput.input2[mpiInput.displacement[i]],
                 mpiInput.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 &requestRootSend[1]);
      MPI_Waitall(2, requestRootSend, MPI_STATUS_IGNORE);
    }
    endTimeRootSend = MPI_Wtime();
  }

  if (mpiInput.rank)  //Only for Workers
  {
    MPI_Irecv(&mpiInput.vectorWorkers1[0],
              mpiInput.numberToSend[mpiInput.rank],
              MPI_FLOAT,
              root,
              0,
              MPI_COMM_WORLD,
              &requestWorkerRecv[0]);
    MPI_Irecv(&mpiInput.vectorWorkers2[0],
              mpiInput.numberToSend[mpiInput.rank],
              MPI_FLOAT,
              root,
              0,
              MPI_COMM_WORLD,
              &requestWorkerRecv[1]);

    MPI_Waitall(2, requestWorkerRecv, MPI_STATUS_IGNORE);
    for (long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++) {
      mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
    }
    MPI_Issend(&mpiInput.vectorWorkers1[0],
               mpiInput.numberToSend[mpiInput.rank],
               MPI_FLOAT,
               root,
               0,
               MPI_COMM_WORLD,
               &requestWorkerSend);  //Tag is 0
    MPI_Wait(&requestWorkerSend, MPI_STATUS_IGNORE);
  }

  if (!mpiInput.rank)  //Only for root
  {
    startTimeRootRecv = MPI_Wtime();
    for (int i = 1; i < mpiInput.num_procs; i++) {
      MPI_Irecv(&mpiInput.output[mpiInput.displacement[i]],
                mpiInput.numberToSend[i],
                MPI_FLOAT,
                i,
                0,
                MPI_COMM_WORLD,
                &requestRootRecv);
      MPI_Wait(&requestRootRecv, MPI_STATUS_IGNORE);
    }
    endTimeRootRecv = MPI_Wtime();

    checkingResultsPrintout(mpiInput.reference,
                            mpiInput.output,
                            mpiInput.workSplit,
                            mpiInput.displacement,
                            mpiInput.numberToSend);  //Only root print out the results.
    returnValue.first = (endTimeRootSend - startTimeRootSend) * 1000;
    returnValue.second = (endTimeRootRecv - startTimeRootRecv) * 1000;
  }
  return returnValue;
}
const std::pair<float, float> blockSend(MPIData& mpiInput) {
  std::pair<float, float> returnValue;
  double startTimeRootSend = 0;
  double endTimeRootSend = 0;
  double startTimeRootRecv = 0;
  double endTimeRootRecv = 0;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  if (!mpiInput.rank)  //Only for root
  {
    // std::cout << "\n\t\tBlocking Send and Receive " << std::endl;
    startTimeRootSend = MPI_Wtime();
    for (int i = 1; i < mpiInput.num_procs; i++) {
      MPI_Send(&mpiInput.input1[mpiInput.displacement[i]],
               mpiInput.numberToSend[i],
               MPI_FLOAT,
               i,
               0,
               MPI_COMM_WORLD);  //Tag is 0
      MPI_Send(&mpiInput.input2[mpiInput.displacement[i]], mpiInput.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }
    endTimeRootSend = MPI_Wtime();
  }

  if (mpiInput.rank)  //Only for Workers
  {
    MPI_Recv(&mpiInput.vectorWorkers1[0],
             mpiInput.numberToSend[mpiInput.rank],
             MPI_FLOAT,
             root,
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(&mpiInput.vectorWorkers2[0],
             mpiInput.numberToSend[mpiInput.rank],
             MPI_FLOAT,
             root,
             0,
             MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    for (long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++) {
      mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
    }
    MPI_Send(&mpiInput.vectorWorkers1[0],
             mpiInput.numberToSend[mpiInput.rank],
             MPI_FLOAT,
             root,
             0,
             MPI_COMM_WORLD);  //Tag is 0
  }

  if (!mpiInput.rank)  //Only for root
  {
    startTimeRootRecv = MPI_Wtime();
    for (int i = 1; i < mpiInput.num_procs; i++) {
      MPI_Recv(&mpiInput.output[mpiInput.displacement[i]],
               mpiInput.numberToSend[i],
               MPI_FLOAT,
               i,
               0,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    endTimeRootRecv = MPI_Wtime();

    checkingResultsPrintout(mpiInput.reference,
                            mpiInput.output,
                            mpiInput.workSplit,
                            mpiInput.displacement,
                            mpiInput.numberToSend);  //Only root print out the results.
    returnValue.first = (endTimeRootSend - startTimeRootSend) * 1000;
    returnValue.second = (endTimeRootRecv - startTimeRootRecv) * 1000;
  }
  return returnValue;
}
const std::pair<float, float> multiNonBlockSend(MPIData& mpiInput) {
  std::pair<float, float> returnValue;
  int lastPointCount = 0;
  double startTimeRootSend = 0;
  double endTimeRootSend = 0;
  double startTimeRootRecv = 0;
  double endTimeRootRecv = 0;

  MPI_Request requestRootSend[2];
  MPI_Request requestRootRecv;
  MPI_Request requestWorkerSend;
  MPI_Request requestWorkerRecv[2];

  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  if (!mpiInput.rank)  //Only for root
  {
    // std::cout << "\n\t\tNon-Blocking Send and Receive with Multiple Tasks" << std::endl;
    int flage = 0;  //set operation to processed.
    startTimeRootSend = MPI_Wtime();
    for (int i = 1; i < mpiInput.num_procs; i++) {
      MPI_Issend(&mpiInput.input1[mpiInput.displacement[i]],
                 mpiInput.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 &requestRootSend[0]);  //Tag is 0
      MPI_Issend(&mpiInput.input2[mpiInput.displacement[i]],
                 mpiInput.numberToSend[i],
                 MPI_FLOAT,
                 i,
                 0,
                 MPI_COMM_WORLD,
                 &requestRootSend[1]);
      do {
        MPI_Testall(2, requestRootSend, &flage, MPI_STATUS_IGNORE);  //2 for two requests above. Check on flage.
        for (; lastPointCount < size && !flage;
             lastPointCount++)  //do the summing while waiting for the sending request is done.
        {
          mpiInput.reference[lastPointCount] = mpiInput.input1[lastPointCount] + mpiInput.input2[lastPointCount];

          MPI_Testall(2, requestRootSend, &flage, MPI_STATUS_IGNORE);  //2 for two requests above. Check on flage.
        }
      } while (!flage);
    }
    endTimeRootSend = MPI_Wtime();
  }

  if (mpiInput.rank)  //Only for Workers
  {
    MPI_Irecv(&mpiInput.vectorWorkers1[0],
              mpiInput.numberToSend[mpiInput.rank],
              MPI_FLOAT,
              root,
              0,
              MPI_COMM_WORLD,
              &requestWorkerRecv[0]);
    MPI_Irecv(&mpiInput.vectorWorkers2[0],
              mpiInput.numberToSend[mpiInput.rank],
              MPI_FLOAT,
              root,
              0,
              MPI_COMM_WORLD,
              &requestWorkerRecv[1]);
    MPI_Waitall(2, requestWorkerRecv, MPI_STATUS_IGNORE);  //2 for two requests above.

    for (long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++) {
      mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
    }
    MPI_Issend(&mpiInput.vectorWorkers1[0],
               mpiInput.numberToSend[mpiInput.rank],
               MPI_FLOAT,
               root,
               0,
               MPI_COMM_WORLD,
               &requestWorkerSend);  //Tag is 0
    MPI_Wait(&requestWorkerSend, MPI_STATUS_IGNORE);
  }

  if (!mpiInput.rank)  //Only for root
  {
    int flage2 = 0;  //set operation to processed.
    startTimeRootRecv = MPI_Wtime();
    for (int i = 1; i < mpiInput.num_procs; i++) {
      MPI_Irecv(&mpiInput.output[mpiInput.displacement[i]],
                mpiInput.numberToSend[i],
                MPI_FLOAT,
                i,
                0,
                MPI_COMM_WORLD,
                &requestRootRecv);
      do {
        MPI_Test(&requestRootRecv, &flage2, MPI_STATUS_IGNORE);  //Check on flage2.
        for (; lastPointCount < size && !flage2;
             lastPointCount++)  //do the summing while waiting for the sending request is done.
        {
          mpiInput.reference[lastPointCount] = mpiInput.input1[lastPointCount] + mpiInput.input2[lastPointCount];

          MPI_Test(&requestRootRecv, &flage2, MPI_STATUS_IGNORE);  //Check on flage.
        }
      } while (!flage2);
    }
    endTimeRootRecv = MPI_Wtime();
    for (; lastPointCount < size; lastPointCount++) {
      mpiInput.reference[lastPointCount] = mpiInput.input1[lastPointCount] + mpiInput.input2[lastPointCount];
    }
    checkingResultsPrintout(mpiInput.reference,
                            mpiInput.output,
                            mpiInput.workSplit,
                            mpiInput.displacement,
                            mpiInput.numberToSend);  //Only root print out the results.
    returnValue.first = (endTimeRootSend - startTimeRootSend) * 1000;
    returnValue.second = (endTimeRootRecv - startTimeRootRecv) * 1000;
  }
  return returnValue;
}

const std::vector<int> chooseFunction(int toInteger) {
  std::vector<int> digits(0, 0);
  std::vector<int> ERROR(0, 0);

  int digit{1};

  while (toInteger > 0) {
    digit = toInteger % 10;
    if (digit > choices) {
      std::cout << "\n\tError Must be integer Argument <= " << choices << std::endl;
      return ERROR;
    }
    digits.push_back(digit);
    toInteger /= 10;
  }
  std::reverse(digits.begin(), digits.end());
  return digits;
}

void compare(const std::vector<std::pair<float, float>>& timing,
             int choices,
             const std::vector<int>& digits,
             int runNumber) {
  std::cout.setf(std::ios::fixed, std::ios::floatfield);

  int j{0};
  int k{0};
  for (long unsigned int i = 0; i < timing.size(); ++i) {
    if (timing[i].first) {
      switch (i) {
        case 0:
          std::cout << "\n\t\t(1) Non-Blocking Scatter " << std::endl;
          break;
        case 1:
          std::cout << "\n\t\t(2) Blocking Scatter " << std::endl;
          break;
        case 2:
          std::cout << "\n\t\t(3) Non-Blocking Send and Receive " << std::endl;
          break;
        case 3:
          std::cout << "\n\t\t(4) Blocking Send and Receive " << std::endl;
          break;
        case 4:
          std::cout << "\n\t\t(5) Non-Blocking Send and Receive with Multiple Tasks" << std::endl;
          break;
        default:
          std::cout << "\nSomething went wrong!\n";
      }
    }
  }
  std::cout << "\n\n\t=============================================================";
  std::cout << "\n\t|| Func ||  Scatter/Send ||   Gather/Receive  || Number Run||";
  std::cout << "\n\t=============================================================";
  for (long unsigned int i = 0; i < timing.size(); ++i) {
    if (timing[i].first) {
      if (k < j) {
        std::cout << "\n\t------------------------------------------------------------";
      }
      std::cout.flags(std::ios::fixed | std::ios::showpoint);
      std::cout.precision(precision);
      std::cout << "\n\t||  " << std::setw(1) << digits[k] << "   ||     " << std::setw(5) << timing[i].first
                << "    ||        " << std::setw(5) << timing[i].second << "     ||    " << std::setw(3) << runNumber
                << "    ||";
      j += 2;
      ++k;
    }
  }
  std::cout << "\n\t=============================================================\n\n";
}

const std::pair<float, float> returnAverage(const std::pair<float, float> (*mpiFunctions)(MPIData&),
                                            MPIData& mpiInput,
                                            unsigned int runNumber) {
  std::pair<float, float> output;
  for (long unsigned int i = 0; i < runNumber; ++i) {
    auto accum = mpiFunctions(mpiInput);
    output.first += accum.first;
    output.second += accum.second;
  }
  output.first /= runNumber;
  output.second /= runNumber;
  return output;
}