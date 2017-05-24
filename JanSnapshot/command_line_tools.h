#ifndef _command_line_tools_h_
#define _command_line_tools_h_

#include <cstring>
#include <string>

//----------------------------------------------------------------------------------------------------

int cl_error = 0;

//----------------------------------------------------------------------------------------------------

bool TestBoolParameter(int argc, const char **argv, int &argi, const char *tag, bool &param1, bool &param2)
{
	if (strcmp(argv[argi], tag) == 0)
	{
		if (argi < argc - 1)
		{
			argi++;
			param1 = param2 = atoi(argv[argi]);
		} else {
			printf("ERROR: option '%s' requires an argument.\n", tag);
			cl_error = 1;
		}

		return true;
	}

	return false;
}

//----------------------------------------------------------------------------------------------------

bool TestBoolParameter(int argc, const char **argv, int &argi, const char *tag, bool &param)
{
	bool fake;
	return TestBoolParameter(argc, argv, argi, tag, param, fake);
}

//----------------------------------------------------------------------------------------------------

bool TestUIntParameter(int argc, const char **argv, int &argi, const char *tag, unsigned int &param)
{
	if (strcmp(argv[argi], tag) == 0)
	{
		if (argi < argc - 1)
		{
			argi++;
			param = (int) atof(argv[argi]);
		} else {
			printf("ERROR: option '%s' requires an argument.\n", tag);
			cl_error = 1;
		}

		return true;
	}

	return false;
}

//----------------------------------------------------------------------------------------------------

bool TestDoubleParameter(int argc, const char **argv, int &argi, const char *tag, double &param1, double &param2)
{
	if (strcmp(argv[argi], tag) == 0)
	{
		if (argi < argc - 1)
		{
			argi++;
			param1 = param2 = atof(argv[argi]);
		} else {
			printf("ERROR: option '%s' requires an argument.\n", tag);
			cl_error = 1;
		}

		return true;
	}

	return false;
}

//----------------------------------------------------------------------------------------------------

bool TestDoubleParameter(int argc, const char **argv, int &argi, const char *tag, double &param)
{
	double fake;
	return TestDoubleParameter(argc, argv, argi, tag, param, fake);
}

//----------------------------------------------------------------------------------------------------

bool TestStringParameter(int argc, const char **argv, int &argi, const char *tag, std::string &param1, std::string &param2)
{
	if (strcmp(argv[argi], tag) == 0)
	{
		if (argi < argc - 1)
		{
			argi++;
			param1 = param2 = argv[argi];
		} else {
			printf("ERROR: option '%s' requires an argument.\n", tag);
			cl_error = 1;
		}

		return true;
	}

	return false;
}

//----------------------------------------------------------------------------------------------------

bool TestStringParameter(int argc, const char **argv, int &argi, const char *tag, std::string &param)
{
	std::string fake;
	return TestStringParameter(argc, argv, argi, tag, param, fake);
}

#endif
