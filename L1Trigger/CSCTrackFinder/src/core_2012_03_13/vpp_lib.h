
void readmemh(std::string fname, ULLONG* mem, ULLONG astart, ULLONG afinish)
{
	std::ifstream ifs (fname.c_str() , std::ifstream::in);
	ULLONG val;
	for (ULLONG i = astart; i <= afinish; i++)
	{
		if (ifs >> std::hex >> val)
		{
			mem[i] = val;	
		}
		else
		{
			std::cerr << "Cannot read file: " << fname << ", addr: " << i << std::endl;
			ifs.close();
			return;
		}
	}
	ifs.close();
	//std::cout << "LUT file: " << fname << " read out successfully." << std::endl;
}
