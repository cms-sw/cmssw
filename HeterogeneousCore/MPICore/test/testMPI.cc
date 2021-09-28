#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <utility>
#include <mpi.h>

    // std::cout << "\n\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    // std::cout << "\n\t+ (1)  Non Blocking Scatter.                             +";
    // std::cout << "\n\t+ (2)  Blocking Scatter.                                 +";
    // std::cout << "\n\t+ (3)  Non Blocking Send and Receive.                    +";
    // std::cout << "\n\t+ (4)  Blocking Send and Receive.                        +";
    // std::cout << "\n\t+ (5)  Non Blocking Send and Receive with Multiple Tasks +";
    // std::cout << "\n\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";

struct MPIinputs
{
    int num_procs{0};
    int rank{0};
    
    std::pair <int,int> workSplit;
    std::vector <float> vect1; //declare vector 1.
    std::vector <float> vect2; //declare vector 2.
    std::vector <float> vect3; //declare vector fulled only by Root to get result from workers.
    std::vector <float> vectRoot; //declare vector to verify the ruslt form each process.
    std::vector <float> vectorWorkers1; //declare vector 1 for workers only.
    std::vector <float> vectorWorkers2; //declare vector 2 for workers only.
    std::vector <int> displacement; //declare vector for selecting location of each element to be sent.
    std::vector <int> numberToSend;


};

int val = 5; //number of functions of MPI.
int sizeOfVector = 21; //default size of vectors.
int precisions = 4; //default digits after decimal point.
int function = 5; //
int Root = 0;
int choice = 0; //user Choice to select function to run.

std::vector <int> userChoices(1,1); //save convertion integer to vector.

const std::vector <int> chooseFunction(int toInteger);//Convert integers to a vector.
std::vector<std::pair <float, float>> timing(val,std::make_pair(0, 0)); //to save time of scatter/send and gather/receive for each function.

void randomGenerator(std::vector<float> &vect); //generate uniform random numbers.
std::pair <int,int> splitProcess(int works, int numberOfProcess); //calcualte for each process number of works.
const std::vector <int> numberDataSend(int numberOfProcess, std::pair<int, int> splitWorks); //findout number of data to be sent for each process.
const std::vector <int> displacmentData(int numberOfProcess, std::pair<int, int> splitWorks, const std::vector <int>& numberDataSend); //findout the index of data to be sent for each process 
void checkingResultsPrintout(std::vector<float> &vectRoot, std::vector<float> &vect3, std::pair <int,int> workSplit, const std::vector<int>& displacement,  const std::vector<int>& numberDataSend);

const std::pair <float, float> nonBlockScatter(MPIinputs& mpiInput);
const std::pair <float, float> blockScatter(MPIinputs& mpiInput);
const std::pair <float, float> nonBlockSend(MPIinputs& mpiInput);
const std::pair <float, float> blockSend(MPIinputs& mpiInput);
const std::pair <float, float> multiNonBlockSend(MPIinputs& mpiInput);

void compare(const std::vector<std::pair <float, float>>& timing, int val, const std::vector <int> digits); //to printout the time for each function that user chose.

int main(int argc, char *argv[]) 
{ 
    if (argc == 2) {
        try{
            sizeOfVector = std::stoll (argv[1], nullptr, 0);
        }
        catch(std::exception& err)
        {
            std::cout << "\n\tError Must be integer Argument!";
            std::cout << "\n\t" << err.what() << std::endl;
            return 0;
        }
    }
    else if(argc > 2)
    {
        try{
            sizeOfVector = std::stoll (argv[1], nullptr, 0);
            choice = std::stoll (argv[2], nullptr, 0);
            userChoices = chooseFunction(choice);
        }
        catch(std::exception& err)
        {
            std::cout << "\n\tError Must be integer Argument!";
            std::cout << "\n\t" << err.what() << std::endl;
            return 0;
        }
    }

    MPIinputs mpiInputs; //greate object from structur to pass into MPI functios.

    
    MPI_Init(&argc, &argv); //initialize communicator environment.
    mpiInputs.num_procs = MPI::COMM_WORLD.Get_size(); //get total size of processes.
    mpiInputs.rank = MPI::COMM_WORLD.Get_rank(); //get each process number.

    mpiInputs.vect1.resize(sizeOfVector); //initialize size.
    mpiInputs.vect2.resize(sizeOfVector);
    mpiInputs.vect3.resize(sizeOfVector);
    mpiInputs.vectRoot.resize(sizeOfVector);

    mpiInputs.workSplit = splitProcess(sizeOfVector,mpiInputs.num_procs);

    if(!mpiInputs.workSplit.first)
    {
         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); 
         return 0;
    }

    mpiInputs.numberToSend = numberDataSend(mpiInputs.num_procs,mpiInputs.workSplit);
    mpiInputs.displacement = displacmentData(mpiInputs.num_procs,mpiInputs.workSplit,mpiInputs.numberToSend);
     
    mpiInputs.vectorWorkers1.resize(mpiInputs.numberToSend[mpiInputs.rank]); //Resizing each process with appropriate Receiving Data.
    mpiInputs.vectorWorkers2.resize(mpiInputs.numberToSend[mpiInputs.rank]);

    if (!mpiInputs.rank) //Only for Root
    {
        randomGenerator(mpiInputs.vect1); //generate random floating numbers from(0,1) Only in the Root.
        randomGenerator(mpiInputs.vect2);
        std::cout << "\n\tNumber of Processes " << mpiInputs.num_procs << std::endl;
        std::cout << "\tNumber of workSplit First " << mpiInputs.workSplit.first << std::endl;
        std::cout << "\tNumber of workSplit Second " << mpiInputs.workSplit.second << std::endl;
        for(int j = 0; j < sizeOfVector; j++)
        {
            mpiInputs.vectRoot[j] = mpiInputs.vect1[j] + mpiInputs.vect2[j]; //Summing for verification.
        }
    }


    for (long unsigned int i = 0; i < userChoices.size(); ++i)
    {
        if (userChoices[i] == 1)
            timing[0] = nonBlockScatter(mpiInputs);
        else if (userChoices[i] == 2)
            timing[1] = blockScatter(mpiInputs);
        else if (userChoices[i] == 3)
            timing[2] = nonBlockSend(mpiInputs);
        else if (userChoices[i] == 4)
            timing[3] = blockSend(mpiInputs);
        else if (userChoices[i] == 5)
            timing[4] = multiNonBlockSend(mpiInputs);
        else
            break;
    }
    

    if(!mpiInputs.rank)
    {
        compare(timing, val,userChoices);
    }
 
    MPI::Finalize();

    return 0;
}

void randomGenerator(std::vector<float> &vect)
{
	
    std::random_device rand;
    std::default_random_engine gener(rand());
    std::uniform_real_distribution<> dis(0., 1.);
	int size = vect.size();
	for (int i = 0; i < size; i++)
	{
	    vect.at(i) = dis(gener);
		
    }

}

std::pair <int,int> splitProcess(int works, int numberOfProcess)
{
    std::pair <int,int> Return{0,0};
    if(numberOfProcess > 1 && numberOfProcess <= works)
    {
        Return.first = works / (numberOfProcess-1); //number of cycle for each process.
        Return.second= works % (numberOfProcess-1); //extra cycle for process.
    }
    else{
        std::cout << "\tError Either No worker are found OR Number Processes Larger than Length!!!\n";
    }
   
    return Return;
}
const std::vector <int> numberDataSend(int numberOfProcess, std::pair<int, int> splitWorks)
{

	std::vector <int> dataSend(numberOfProcess, splitWorks.first);
	dataSend[0] = 0;
	for (int i = 1; i < splitWorks.second+1; i++) //neglect root 
	{
		dataSend[i] += 1; //extra work for each first processes.
	}
	return dataSend;
	
}
const std::vector <int> displacmentData(int numberOfProcess, std::pair<int, int> splitWorks, const std::vector <int>& numberDataSend)
{

	std::vector <int> displacment(numberOfProcess, splitWorks.first);

    
	displacment[0] = 0;
	displacment[1] = 0; //start Here.
    
	for (int i = 2; i < numberOfProcess; i++) //neglect root 
	{
		displacment[i] = numberDataSend[i-1] + displacment[i-1]; //extra work for each first processes.
	}
	return displacment;

}

void checkingResultsPrintout(std::vector<float> &vectRoot, std::vector<float> &vect3, std::pair <int,int> workSplit, const std::vector<int>& displacement,  const std::vector<int>& numberDataSend)
{
    float percent  {0.0};
    float totalError { 0.0};
    int p {1};
    for(int j = 0; j < sizeOfVector; j++)
    {
        percent = ((vectRoot[j] - vect3[j]) / vectRoot[j])*100;
        totalError += percent;
       
    }
    if(totalError)
    {
        std::cout << "\n-------------------------------------------------------\n";
        std::cout << "| RootSum | WorksSum | Error   | Error %  | Process # |";
        std::cout << "\n-------------------------------------------------------\n";
        std::cout.precision(precisions);
        for(int j = 0; j < sizeOfVector; j++)
        {
           
            std::cout << "| " << vectRoot[j] << "  | " << vect3[j] << "  |" << std::setw(9) << vectRoot[j] - vect3[j] << " |"<< std::setw(9)  << percent << " |" << std::setw(9) << p << " |\n";
            
            if(j+1  == displacement[p+1])
            {
                ++p;
            }
        }
        std::cout << "-------------------------------------------------------\n";
        std::cout << "-Total Error is " << totalError << std::endl;
        for(long unsigned int j = 1; j <displacement.size(); j++)
        {
            std::cout <<  "Process [" << j << "]" << " Worked On " << numberDataSend [j] << " Data\n";
        }
    }

    
}


const std::pair <float, float> nonBlockScatter(MPIinputs& mpiInput)
{
    std::pair <float, float> Retrun;
    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    double startTimeScatter = 0;
    double endTimeScatter = 0;
    double startTimeGather = 0;
    double endTimeGather = 0;
    
    
    MPI_Request requestRootScatter[2];
    MPI_Request requestRootGather;

    if(!mpiInput.rank)
        std::cout << "\n\t\tNon-Blocking Scatter " << std::endl;
    startTimeScatter = MPI_Wtime(); //get time before scattering.

    //Non-Blocking Scatter.
    MPI_Iscatterv(&mpiInput.vect1[0],&mpiInput.numberToSend[0],&mpiInput.displacement[0],MPI_FLOAT,&mpiInput.vectorWorkers1[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,0,MPI_COMM_WORLD,&requestRootScatter[0]);
    MPI_Iscatterv(&mpiInput.vect2[0],&mpiInput.numberToSend[0],&mpiInput.displacement[0],MPI_FLOAT,&mpiInput.vectorWorkers2[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,0,MPI_COMM_WORLD,&requestRootScatter[1]);
    MPI_Waitall(2,requestRootScatter, MPI_STATUS_IGNORE);

    endTimeScatter = MPI_Wtime(); //get time after scattering.
   

    if(mpiInput.rank)//Only for Workers 
    {
        for(long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++)
        {
            mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
        }
        
    }

    startTimeGather = MPI_Wtime(); //get time before Gathering.

    //Non Blocking Gathering.
    MPI_Igatherv(&mpiInput.vectorWorkers1[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,&mpiInput.vect3[0],&mpiInput.numberToSend[0],&mpiInput.displacement[0],MPI_FLOAT,0,MPI_COMM_WORLD,&requestRootGather);
    
    MPI_Wait(&requestRootGather, MPI_STATUS_IGNORE);
    endTimeGather = MPI_Wtime(); //get time after Gathering.

    

        
    

    if(!mpiInput.rank) //Only Root print out the results.
    {
        checkingResultsPrintout(mpiInput.vectRoot,mpiInput.vect3, mpiInput.workSplit, mpiInput.displacement, mpiInput.numberToSend);
        Retrun.first = (endTimeScatter - startTimeScatter)*1000;
        Retrun.second = (endTimeGather - startTimeGather)*1000;
        std::cout << "\nScattreing Time [" << mpiInput.rank << "]" << " = " << Retrun.first << " ms";
        std::cout << "\nGathering Time [" << mpiInput.rank << "]" << " = " << Retrun.second << " ms\n";
        
    }
    return Retrun;
    
}
const std::pair <float, float> blockScatter(MPIinputs& mpiInput)
{
    std::pair <float, float> Retrun;
    std::cout.setf(std::ios::fixed,std::ios::floatfield);

    double startTimeScatter = 0;
    double endTimeScatter = 0;
    double startTimeGather = 0;
    double endTimeGather = 0;

    //MPI_Request requestRoot;

    if(!mpiInput.rank)
        std::cout << "\n\t\tBlocking Scatter " << std::endl;
    
    //Blocking Scattering.
    mpiInput.vectorWorkers1.resize(mpiInput.numberToSend[mpiInput.rank]); //Resizing each process with appropriate Receiving Data.
    mpiInput.vectorWorkers2.resize(mpiInput.numberToSend[mpiInput.rank]);
    
    startTimeScatter =  MPI_Wtime();
    MPI_Scatterv(&mpiInput.vect1[0],&mpiInput.numberToSend[0],&mpiInput.displacement[0],MPI_FLOAT,&mpiInput.vectorWorkers1[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Scatterv(&mpiInput.vect2[0],&mpiInput.numberToSend[0],&mpiInput.displacement[0],MPI_FLOAT,&mpiInput.vectorWorkers2[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,0,MPI_COMM_WORLD);
    endTimeScatter =  MPI_Wtime();

    if(mpiInput.rank)//Only for Workers 
    {
        for(long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++)
        {
            mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
        }
        
    }

    startTimeGather = MPI_Wtime();
    //Blocking Gathering.
    MPI_Gatherv(&mpiInput.vectorWorkers1[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,&mpiInput.vect3[0],&mpiInput.numberToSend[0],&mpiInput.displacement[0],MPI_FLOAT,0,MPI_COMM_WORLD);

    endTimeGather = MPI_Wtime();
    

    

    if(!mpiInput.rank) //Only Root print out the results.
    {
        checkingResultsPrintout(mpiInput.vectRoot,mpiInput.vect3, mpiInput.workSplit, mpiInput.displacement, mpiInput.numberToSend);
        Retrun.first = (endTimeScatter - startTimeScatter)*1000;
        Retrun.second = (endTimeGather - startTimeGather)*1000;
        std::cout << "\nScattreing Time [" << mpiInput.rank << "]" << " = " << Retrun.first << " ms";
        std::cout << "\nGathering Time [" << mpiInput.rank << "]" << " = " << Retrun.second << " ms\n";       
        
    }
   return Retrun;
}
const std::pair <float, float> nonBlockSend(MPIinputs& mpiInput)
{
    std::pair <float, float> Retrun;
    double startTimeRootSend = 0;
    double endTimeRootSend = 0;
    double startTimeRootRecv = 0;
    double endTimeRootRecv = 0;

    MPI_Request requestRootSend[2];
    MPI_Request requestRootRecv;
    MPI_Request requestWorkerSend;
    MPI_Request requestWorkerRecv[1];

    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    
    

    if (!mpiInput.rank) //Only for Root
    {
        std::cout << "\n\t\tNon-Blocking Send and Receive " << std::endl;
        startTimeRootSend =  MPI_Wtime();
        for(int i = 1; i < mpiInput.num_procs; i++)
        {
            MPI_Issend(&mpiInput.vect1[mpiInput.displacement[i]], mpiInput.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestRootSend[0]); //Tag is 0
            MPI_Issend(&mpiInput.vect2[mpiInput.displacement[i]], mpiInput.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestRootSend[1]);
            MPI_Waitall(2,requestRootSend, MPI_STATUS_IGNORE);
        }
        endTimeRootSend =  MPI_Wtime();
    }


    if(mpiInput.rank)//Only for Workers 
    {
        MPI_Irecv(&mpiInput.vectorWorkers1[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,Root,0,MPI_COMM_WORLD, &requestWorkerRecv[0]);
        MPI_Irecv(&mpiInput.vectorWorkers2[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,Root,0,MPI_COMM_WORLD, &requestWorkerRecv[1]);
    
        MPI_Waitall(2,requestWorkerRecv,MPI_STATUS_IGNORE);
        for(long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++)
        {
            mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
        }
        MPI_Issend(&mpiInput.vectorWorkers1[0], mpiInput.numberToSend[mpiInput.rank], MPI_FLOAT, Root, 0, MPI_COMM_WORLD, &requestWorkerSend); //Tag is 0
        MPI_Wait(&requestWorkerSend,MPI_STATUS_IGNORE);
    }


    if(!mpiInput.rank)//Only for Root
    {
        startTimeRootRecv=  MPI_Wtime();
        for(int i = 1; i < mpiInput.num_procs; i++)
        {
            MPI_Irecv(&mpiInput.vect3[mpiInput.displacement[i]],mpiInput.numberToSend[i],MPI_FLOAT,i,0,MPI_COMM_WORLD, &requestRootRecv);
            MPI_Wait(&requestRootRecv,MPI_STATUS_IGNORE);
        }
        endTimeRootRecv=  MPI_Wtime();
        
        checkingResultsPrintout(mpiInput.vectRoot,mpiInput.vect3, mpiInput.workSplit, mpiInput.displacement, mpiInput.numberToSend); //Only Root print out the results.
        Retrun.first = (endTimeRootSend - startTimeRootSend)*1000;
        Retrun.second = (endTimeRootRecv - startTimeRootRecv)*1000;
        std::cout << "\nTime Sending Root = " << Retrun.first << " ms";
        std::cout << "\nTime Receiving Root = " << Retrun.second << " ms\n";
    }
    return Retrun;
}
const std::pair <float, float> blockSend(MPIinputs& mpiInput)
{
    std::pair <float, float> Retrun;
    double startTimeRootSend = 0;
    double endTimeRootSend = 0;
    double startTimeRootRecv = 0;
    double endTimeRootRecv = 0;

    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    
    if (!mpiInput.rank) //Only for Root
    {
       
        std::cout << "\n\t\tBlocking Send and Receive " << std::endl;
        startTimeRootSend =  MPI_Wtime();
        for(int i = 1; i < mpiInput.num_procs; i++)
        {
            MPI_Send(&mpiInput.vect1[mpiInput.displacement[i]], mpiInput.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD); //Tag is 0
            MPI_Send(&mpiInput.vect2[mpiInput.displacement[i]], mpiInput.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        endTimeRootSend =  MPI_Wtime();
    }


    if(mpiInput.rank)//Only for Workers 
    {
        MPI_Recv(&mpiInput.vectorWorkers1[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,Root,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&mpiInput.vectorWorkers2[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,Root,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
       
        for(long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++)
        {
            mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
        }
        MPI_Send(&mpiInput.vectorWorkers1[0], mpiInput.numberToSend[mpiInput.rank], MPI_FLOAT, Root, 0, MPI_COMM_WORLD); //Tag is 0
    }


    if(!mpiInput.rank)//Only for Root
    {
        startTimeRootRecv=  MPI_Wtime();
        for(int i = 1; i < mpiInput.num_procs; i++)
        {
            MPI_Recv(&mpiInput.vect3[mpiInput.displacement[i]],mpiInput.numberToSend[i],MPI_FLOAT,i,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        endTimeRootRecv=  MPI_Wtime();
       
        checkingResultsPrintout(mpiInput.vectRoot,mpiInput.vect3, mpiInput.workSplit, mpiInput.displacement, mpiInput.numberToSend); //Only Root print out the results.
        Retrun.first = (endTimeRootSend - startTimeRootSend)*1000;
        Retrun.second = (endTimeRootRecv - startTimeRootRecv)*1000;
        std::cout << "\nTime Sending Root = " << Retrun.first << " ms";
        std::cout << "\nTime Receiving Root = " << Retrun.second << " ms\n";

    }
    return Retrun;
}
const std::pair <float, float> multiNonBlockSend(MPIinputs& mpiInput)
{
    std::pair <float, float> Retrun;
    int lastPointCount = 0;
    double startTimeRootSend = 0;
    double endTimeRootSend = 0;
    double startTimeRootRecv = 0;
    double endTimeRootRecv = 0;

 
    MPI_Request requestRootSend[2];
    MPI_Request requestRootRecv;
    MPI_Request requestWorkerSend;
    MPI_Request requestWorkerRecv[2];
    
    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    

    if (!mpiInput.rank) //Only for Root
    {
        std::cout << "\n\t\tNon-Blocking Send and Receive with Multiple Tasks" << std::endl;
        int flage = 0; //set operation to processed.
        startTimeRootSend =  MPI_Wtime();
        for(int i = 1; i < mpiInput.num_procs; i++)
        {
            MPI_Issend(&mpiInput.vect1[mpiInput.displacement[i]], mpiInput.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestRootSend[0]); //Tag is 0
            MPI_Issend(&mpiInput.vect2[mpiInput.displacement[i]], mpiInput.numberToSend[i], MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestRootSend[1]);
            do{
                MPI_Testall(2,requestRootSend,&flage, MPI_STATUS_IGNORE); //2 for two requests above. Check on flage.
                for(; lastPointCount < sizeOfVector && !flage; lastPointCount++) //do the summing while waiting for the sending request is done.
                {
                    mpiInput.vectRoot[lastPointCount] = mpiInput.vect1[lastPointCount] + mpiInput.vect2[lastPointCount];

                    MPI_Testall(2,requestRootSend,&flage, MPI_STATUS_IGNORE); //2 for two requests above. Check on flage.
                }
            }while(!flage);
        }
        endTimeRootSend =  MPI_Wtime();
    }

    
    if(mpiInput.rank)//Only for Workers 
    {
        MPI_Irecv(&mpiInput.vectorWorkers1[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,Root,0,MPI_COMM_WORLD, &requestWorkerRecv[0]);
        MPI_Irecv(&mpiInput.vectorWorkers2[0],mpiInput.numberToSend[mpiInput.rank],MPI_FLOAT,Root,0,MPI_COMM_WORLD, &requestWorkerRecv[1]);
        MPI_Waitall(2,requestWorkerRecv,MPI_STATUS_IGNORE);//2 for two requests above.
        
        for(long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++)
        {
            mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
        }
        MPI_Issend(&mpiInput.vectorWorkers1[0], mpiInput.numberToSend[mpiInput.rank], MPI_FLOAT, Root, 0, MPI_COMM_WORLD, &requestWorkerSend); //Tag is 0
        MPI_Wait(&requestWorkerSend,MPI_STATUS_IGNORE);
    }


    if(!mpiInput.rank)//Only for Root
    {
        int flage2 = 0; //set operation to processed.
        startTimeRootRecv = MPI_Wtime();
        for(int i = 1; i < mpiInput.num_procs; i++)
        {
            MPI_Irecv(&mpiInput.vect3[mpiInput.displacement[i]],mpiInput.numberToSend[i],MPI_FLOAT,i,0,MPI_COMM_WORLD, &requestRootRecv);
            do{
                MPI_Test(&requestRootRecv,&flage2,MPI_STATUS_IGNORE);//Check on flage2.
                for(; lastPointCount < sizeOfVector && !flage2; lastPointCount++) //do the summing while waiting for the sending request is done.
                {
                    mpiInput.vectRoot[lastPointCount] = mpiInput.vect1[lastPointCount] + mpiInput.vect2[lastPointCount];

                    MPI_Test(&requestRootRecv,&flage2,MPI_STATUS_IGNORE);//Check on flage.
                }
            }while(!flage2);

        }
        endTimeRootRecv = MPI_Wtime();
        for(; lastPointCount < sizeOfVector; lastPointCount++)
        {
            
            mpiInput.vectRoot[lastPointCount] = mpiInput.vect1[lastPointCount] + mpiInput.vect2[lastPointCount];
        }
        checkingResultsPrintout(mpiInput.vectRoot,mpiInput.vect3, mpiInput.workSplit, mpiInput.displacement, mpiInput.numberToSend); //Only Root print out the results.
        Retrun.first = (endTimeRootSend - startTimeRootSend)*1000;
        Retrun.second = (endTimeRootRecv - startTimeRootRecv)*1000;
        std::cout << "\nTime Sending Root = " << Retrun.first << " ms";
        std::cout << "\nTime Receiving Root = " << Retrun.second << " ms\n";
        
    }
    return Retrun;
}


const std::vector <int> chooseFunction(int toInteger)
{
    std::vector <int> digits(0,0);
    std::vector <int> ERROR(0,0);
  
    int copyTointger{0};
    int divition{ 0 };
    int length{0};


    copyTointger = toInteger;
   
    
    while (copyTointger)
    {
        copyTointger /= 10;
        ++length;
    }

	divition = std::pow(10.0, (float)(length-1));
	copyTointger = toInteger;
	digits.resize(length);

    for (int i = 0; i < length; i++)
    {
		copyTointger = toInteger / divition;
        if(copyTointger > val)
        {
            std::cout << "\n\tError Must be integer Argument <= " << val << std::endl;
            return ERROR;
        }
        digits[i]= copyTointger;
		copyTointger *= divition;
		toInteger -= copyTointger;
		divition /= 10;
    }

    return digits;

}

void compare(const std::vector<std::pair <float, float>> &timing, int val, const std::vector <int> digits)
{
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
    
	int j{ 0 };
    int k{ 0 };
	std::cout << "\n\n\t===================================================";
	std::cout << "\n\t||      ||  Scatter/Send   ||   Gather/Receive   ||";
	std::cout << "\n\t===================================================";
	for (long unsigned int i = 0; i < timing.size(); i++)
	{
		
		if (timing[i].first)
		{
			if (k < j)
			{
				std::cout << "\n\t---------------------------------------------------";

			}
            std::cout << std::fixed;
            std::setprecision(4);
			std::cout << "\n\t|| " << std::setw(2) << digits[k] << "   ||     " << std::setw(5) << timing[i].first << "    ||        " << std::setw(5) << timing[i].second << "     ||";
			j+=2;
            ++k;
		}
	}
	std::cout << "\n\t===================================================\n\n";
}