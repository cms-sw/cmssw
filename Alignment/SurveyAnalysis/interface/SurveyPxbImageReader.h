#ifndef GUARD_surveypxbimagereader_h
#define GUARD_surveypxbimagereader_h

#include <sstream>
#include <vector>
#include <fstream>
#include <string>

//! Class to hold one picture of the BPix survey
template <class T> class SurveyPxbImageReader
{
    public:
		typedef std::vector<T> measurements_t;

	// Constructors
	//! Empty default constructor
	SurveyPxbImageReader() {};
	//! Constructor with filename and destination vector
	SurveyPxbImageReader(std::string filename, measurements_t &measurements, SurveyPxbImage::count_t reserve = 800)
	{ 
		read(filename, measurements, reserve);
   	};
	
	//! Reads a file, parses its content and fills the data vector
	//! All data after a hash sign (#) is treated as a comment and not read
	//! \param filename Filename of the file to be read
	//! \param measurements Vector containing the measurements, previous content will be deleted
	//! \param reserve Initial size of the vector, set with vector::reserve()
	//! \return number of succesfully read entries
	SurveyPxbImage::count_t read(std::string filename, measurements_t &measurements, SurveyPxbImage::count_t reserve = 800)
	{
		std::ifstream infile(filename.c_str());
		if (!infile)
		{
			std::cerr << "Cannot open file " << filename
			     << " - operation aborted." << std::endl;
			return 0;
		}

		// prepare the measurements vector
		measurements.clear();
		measurements.reserve(reserve);

		// container for the current line
		std::string aLine;

	    // loop over lines of input file
	    while (std::getline(infile,aLine))
	    {
	        // strip off everything after a hash
	        std::string stripped = "";
	        std::string::iterator iter = std::find(aLine.begin(), aLine.end(), '#');
	        std::copy(aLine.begin(), iter, std::back_inserter(stripped));
	        // read one measurment and add to vector if successfull
	        std::istringstream iss(stripped, std::istringstream::in);
	        T curMeas(iss);
	        if (curMeas.isValid())
	        {
	             measurements.push_back(curMeas);
	        }
	    }

		return measurements.size();
	}

    protected:

};

#endif

