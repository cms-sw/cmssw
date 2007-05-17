#include <L1TriggerConfig/CSCTFConfigProducers/interface/CSCTFConfigProducer.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h> //21 and 19

#include <stdio.h>
#include <errno.h>
#include <iostream>
#include <fstream>

CSCTFConfigProducer::CSCTFConfigProducer(const edm::ParameterSet& pset) {
	ptLUT_path         = pset.getUntrackedParameter<std::string>("ptLUT_path","");
	dt1LUT_path        = pset.getUntrackedParameter<std::string>("dt1LUT_path","");
	dt2LUT_path        = pset.getUntrackedParameter<std::string>("dt2LUT_path","");
	localPhiLUT_path   = pset.getUntrackedParameter<std::string>("localPhiLUT_path","");
	globalPhi1LUT_path = pset.getUntrackedParameter<std::string>("globalPhi1LUT_path","");
	globalPhi2LUT_path = pset.getUntrackedParameter<std::string>("globalPhi2LUT_path","");
	globalPhi3LUT_path = pset.getUntrackedParameter<std::string>("globalPhi3LUT_path","");
	globalPhi4LUT_path = pset.getUntrackedParameter<std::string>("globalPhi4LUT_path","");
	globalPhi5LUT_path = pset.getUntrackedParameter<std::string>("globalPhi5LUT_path","");
	globalEta1LUT_path = pset.getUntrackedParameter<std::string>("globalEta1LUT_path","");
	globalEta2LUT_path = pset.getUntrackedParameter<std::string>("globalEta2LUT_path","");
	globalEta3LUT_path = pset.getUntrackedParameter<std::string>("globalEta3LUT_path","");
	globalEta4LUT_path = pset.getUntrackedParameter<std::string>("globalEta4LUT_path","");
	globalEta5LUT_path = pset.getUntrackedParameter<std::string>("globalEta5LUT_path","");
	setWhatProduced(this, &CSCTFConfigProducer::produceL1MuCSCPtLutRcd);
	setWhatProduced(this, &CSCTFConfigProducer::produceL1MuCSCLocalPhiLutRcd);
	setWhatProduced(this, &CSCTFConfigProducer::produceL1MuCSCGlobalLutsRcd);
}

std::auto_ptr<L1MuCSCPtLut> CSCTFConfigProducer::produceL1MuCSCPtLutRcd(const L1MuCSCPtLutRcd& iRecord){
	std::auto_ptr<L1MuCSCPtLut> pt_lut = std::auto_ptr<L1MuCSCPtLut>( new L1MuCSCPtLut() );

	if( ptLUT_path.length() ){
			readLUT(ptLUT_path, (unsigned short *)pt_lut->pt_lut, 1<<21);
	} else {
		// Generating
		//pt_lut->pt_lut[] = 0;
	}

	return pt_lut;
}

std::auto_ptr<L1MuCSCDTLut> CSCTFConfigProducer::produceL1MuCSCDTLutRcd(const L1MuCSCDTLutRcd& iRecord){
	std::auto_ptr<L1MuCSCDTLut> dt_lut = std::auto_ptr<L1MuCSCDTLut>( new L1MuCSCDTLut() );

	if( dt1LUT_path.length() ){
			readLUT(dt1LUT_path, (unsigned short *)dt_lut->dt_lut[0], 1<<19);
	} else {
		// Generating
		//phi_lut->phi_lut[] = 0;
	}
	if( dt2LUT_path.length() ){
			readLUT(dt2LUT_path, (unsigned short *)dt_lut->dt_lut[1], 1<<19);
	} else {
		// Generating
		//phi_lut->phi_lut[] = 0;
	}
	return dt_lut;
}

std::auto_ptr<L1MuCSCLocalPhiLut> CSCTFConfigProducer::produceL1MuCSCLocalPhiLutRcd(const L1MuCSCLocalPhiLutRcd& iRecord){
	std::auto_ptr<L1MuCSCLocalPhiLut> phi_lut = std::auto_ptr<L1MuCSCLocalPhiLut>( new L1MuCSCLocalPhiLut() );

	if( localPhiLUT_path.length() ){
			readLUT(localPhiLUT_path, (unsigned short *)phi_lut->phi_lut, 1<<19);
	} else {
		// Generating
		//phi_lut->phi_lut[] = 0;
	}
	return phi_lut;
}

std::auto_ptr<L1MuCSCGlobalLuts>  CSCTFConfigProducer::produceL1MuCSCGlobalLutsRcd(const L1MuCSCGlobalLutsRcd& iRecord){
	std::auto_ptr<L1MuCSCGlobalLuts> luts = std::auto_ptr<L1MuCSCGlobalLuts>( new L1MuCSCGlobalLuts() );

	if( globalPhi1LUT_path.length() ) readLUT(globalPhi1LUT_path, (unsigned short *)luts->global_phi_lut[0], 1<<19);
	if( globalPhi2LUT_path.length() ) readLUT(globalPhi2LUT_path, (unsigned short *)luts->global_phi_lut[1], 1<<19);
	if( globalPhi3LUT_path.length() ) readLUT(globalPhi3LUT_path, (unsigned short *)luts->global_phi_lut[2], 1<<19);
	if( globalPhi4LUT_path.length() ) readLUT(globalPhi4LUT_path, (unsigned short *)luts->global_phi_lut[3], 1<<19);
	if( globalPhi5LUT_path.length() ) readLUT(globalPhi5LUT_path, (unsigned short *)luts->global_phi_lut[4], 1<<19);

	if( globalEta1LUT_path.length() ) readLUT(globalEta1LUT_path, (unsigned short *)luts->global_eta_lut[0], 1<<19);
	if( globalEta2LUT_path.length() ) readLUT(globalEta2LUT_path, (unsigned short *)luts->global_eta_lut[1], 1<<19);
	if( globalEta3LUT_path.length() ) readLUT(globalEta3LUT_path, (unsigned short *)luts->global_eta_lut[2], 1<<19);
	if( globalEta4LUT_path.length() ) readLUT(globalEta4LUT_path, (unsigned short *)luts->global_eta_lut[3], 1<<19);
	if( globalEta5LUT_path.length() ) readLUT(globalEta5LUT_path, (unsigned short *)luts->global_eta_lut[4], 1<<19);

	return luts;
}

void CSCTFConfigProducer::readLUT(std::string path, unsigned short* lut, unsigned long length){
	// Reading
	if( path.find(".bin") != std::string::npos ) { // Binary format
		std::ifstream file(path.c_str(), std::ios::binary);
		try {
			file.read((char*)lut,length);
		} catch (...) {
			//edm::LogError("CSCTFConfigProducer") << "Cannot read 1<<21 words from the file (incorrect Pt LUT size?)\n";
			throw cms::Exception("Incorrect Pt LUT size?")<<"CSCTFConfigProducer errno="<<errno;
		}
		file.close();
	} else {
		std::ifstream file(path.c_str());
		for(unsigned int address=0; !file.eof() && address<length; address++) //CSCBitWidths::kPtAddressWidth
			file >> lut[address];
		file.close();
	}
}
