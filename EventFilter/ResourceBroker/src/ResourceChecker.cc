/*
 * ResourceChecker.cc
 *
 *  Created on: Nov 23, 2011
 *      Author: aspataru : aspataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/ResourceChecker.h"
#include "EventFilter/Utilities/interface/GlobalEventNumber.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "interface/evb/i2oEVBMsgs.h"
#include "interface/shared/frl_header.h"
#include "interface/shared/fed_header.h"
#include "interface/shared/fed_trailer.h"
#include "EvffedFillerRB.h"

#include <sstream>

using namespace evf;
using std::stringstream;
using std::ostringstream;
using std::hex;
using std::dec;

//______________________________________________________________________________
ResourceChecker::ResourceChecker(FUResource* const resToCheck) :
	res_(resToCheck) {
}

//______________________________________________________________________________
void ResourceChecker::processDataBlock(MemRef_t* bufRef) throw (evf::Exception) {
	// reset iBlock_/nBlock_ counters
	if (res_->iBlock_ == res_->nBlock_) {
		res_->iBlock_ = 0;
		res_->nBlock_ = 0xffffffff;
	}

	I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block =
			(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) bufRef->getDataLocation();

	UInt_t iBlock = block->blockNb;
	UInt_t nBlock = block->nbBlocksInSuperFragment;
	UInt_t iSuperFrag = block->superFragmentNb;
	UInt_t nSuperFrag = block->nbSuperFragmentsInEvent;
	UInt_t fuResourceId = block->fuTransactionId;
	UInt_t buResourceId = block->buResourceId;
	UInt_t evtNumber = block->eventNumber;
	stringstream oss;
	oss << "TransId:" << fuResourceId << " BUResourceId:" << buResourceId
			<< " eventNumber:" << evtNumber << " ";
	// check fuResourceId consistency
	if (fuResourceId != res_->fuResourceId_) {
		res_->nbErrors_++;

		oss << "RU/FU fuResourceId mismatch." << " Received:" << fuResourceId
				<< " Expected:" << res_->fuResourceId_;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	// check iBlock consistency
	if (iBlock != res_->iBlock_) {
		res_->nbErrors_++;
		oss << "RU/FU block number mismatch." << " Received:" << iBlock
				<< " Expected:" << res_->iBlock_;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	// check iSuperFrag consistency
	if (iSuperFrag != res_->iSuperFrag_) {
		res_->nbErrors_++;
		oss << "RU/FU superfragment number mismatch." << " Received:"
				<< iSuperFrag << " Expected:" << res_->iSuperFrag_;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	// assign nBlock_
	if (iBlock == 0) {
		res_->nBlock_ = nBlock;
	} else {
		// check nBlock_
		if (nBlock != res_->nBlock_) {
			res_->nbErrors_++;
			oss << "RU/FU number of blocks mismatch." << " Received:" << nBlock
					<< " Expected:" << res_->nBlock_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}
	}

	// if this is the first block in the event,
	// *assign* evtNumber,buResourceId,nSuperFrag ...
	if (iBlock == 0 && iSuperFrag == 0) {
		res_->evtNumber_ = evtNumber;
		res_->buResourceId_ = buResourceId;
		res_->nSuperFrag_ = nSuperFrag;

		res_->shmCell_->setEvtNumber(evtNumber);
		res_->shmCell_->setBuResourceId(buResourceId);

		// check that buffers are allocated for nSuperFrag superfragments
		if (res_->nSuperFrag_ > res_->nSuperFragMax_) {
			res_->nbErrors_++;
			oss << "Invalid maximum number of superfragments."
					<< " fuResourceId:" << res_->fuResourceId_ << " evtNumber:"
					<< res_->evtNumber_ << " nSuperFrag:" << res_->nSuperFrag_
					<< " nSuperFragMax:" << res_->nSuperFragMax_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}
	}
	// ... otherwise,
	// *check* evtNumber,buResourceId,nSuperFrag
	else {
		// check evtNumber
		if (evtNumber != res_->evtNumber_) {
			res_->nbErrors_++;
			oss << "RU/FU evtNumber mismatch." << " Received:" << evtNumber
					<< " Expected:" << res_->evtNumber_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		// check buResourceId
		if (buResourceId != res_->buResourceId_) {
			res_->nbErrors_++;
			oss << "RU/FU buResourceId mismatch."// implemented in subclasses (for now) << " Received:"
					<< buResourceId << " Expected:" << res_->buResourceId_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		// check nSuperFrag
		if (nSuperFrag != res_->nSuperFrag_) {
			res_->nbErrors_++;
			oss << "RU/FU number of superfragments mismatch." << " Received:"
					<< nSuperFrag << " Expected:" << res_->nSuperFrag_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}
	}

	// check payload
	try {
		checkDataBlockPayload(bufRef);
	} catch (xcept::Exception& e) {
		oss << "data block payload failed check." << " evtNumber:"
				<< res_->evtNumber_ << " buResourceId:" << res_->buResourceId_
				<< " iSuperFrag:" << res_->iSuperFrag_;
		XCEPT_RETHROW(evf::Exception, oss.str(), e);
	}

	res_->appendBlockToSuperFrag(bufRef);

	// increment iBlock_, as expected for the next message
	res_->iBlock_++;

	// superfragment complete ...
	bool lastBlockInSuperFrag = (iBlock == nBlock - 1);
	if (lastBlockInSuperFrag) {

		// ... fill the FED buffers contained in the superfragment
		try {
			// UPDATED
			res_->superFragSize(); // if event exceeds size an exception is thrown here, keep it distinct from SF corruption
		} catch (xcept::Exception& e) {
			oss << "Invalid super fragment size." << " evtNumber:"
					<< res_->evtNumber_ << " buResourceId:"
					<< res_->buResourceId_ << " iSuperFrag:"
					<< res_->iSuperFrag_;
			res_->removeLastAppendedBlockFromSuperFrag();
			XCEPT_RETHROW(evf::Exception, oss.str(), e);
		}
		try {
			res_->fillSuperFragPayload();
			findFEDs();

		} catch (xcept::Exception& e) {
			oss << "Invalid super fragment." << " evtNumber:"
					<< res_->evtNumber_ << " buResourceId:"
					<< res_->buResourceId_ << " iSuperFrag:"
					<< res_->iSuperFrag_;
			res_->removeLastAppendedBlockFromSuperFrag();
			XCEPT_RETHROW(evf::Exception, oss.str(), e);
		}

		// ... release the buffers associated with the superfragment
		try {
			res_->releaseSuperFrag();
		} catch (xcept::Exception& e) {
			res_->nbErrors_++;
			oss << "Failed to release super fragment." << " evtNumber:"
					<< res_->evtNumber_ << " buResourceId:"
					<< res_->buResourceId_ << " iSuperFrag:"
					<< res_->iSuperFrag_;
			XCEPT_RETHROW(evf::Exception, oss.str(), e);
		}

		// increment iSuperFrag_, as expected for the next message(s)
		res_->iSuperFrag_++;

	} // lastBlockInSuperFragment

	return;
}

//______________________________________________________________________________
void ResourceChecker::checkDataBlockPayload(MemRef_t* bufRef)
		throw (evf::Exception) {
	UInt_t frameSize = 0;
	UInt_t bufSize = 0;
	UInt_t segSize = 0;
	UInt_t segSizeExpected = 0;

	frlh_t *frlHeader = 0;

	UChar_t *blockAddr = 0;
	UChar_t *frlHeaderAddr = 0;

	frameSize = sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);

	blockAddr = (UChar_t*) bufRef->getDataLocation();
	frlHeaderAddr = blockAddr + frameSize;
	frlHeader = (frlh_t*) frlHeaderAddr;

	I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block =
			(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) blockAddr;

	// check that FRL trigno is consistent with FU evtNumber
	if (res_->evtNumber_ != frlHeader->trigno) {
		res_->nbErrors_++;
		stringstream oss;
		oss << "FRL header \"trigno\" does not match " << "FU  \"evtNumber\"."
				<< " trigno:" << frlHeader->trigno << " evtNumber:"
				<< res_->evtNumber_;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	// check that FRL trigno is consistent with RU eventNumber
	if (block->eventNumber != frlHeader->trigno) {
		res_->nbErrors_++;
		stringstream oss;
		oss << "FRL header \"trigno\" does not match "
				<< "RU builder header \"eventNumber\"." << " trigno:"
				<< frlHeader->trigno << " eventNumber:" << block->eventNumber;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	// check that block numbers reported by FRL / RU are consistent
	if (block->blockNb != frlHeader->segno) {
		res_->nbErrors_++;
		stringstream oss;
		oss << "FRL header \"segno\" does not match"
				<< "RU builder header \"blockNb\"." << " segno:"
				<< frlHeader->segno << " blockNb:" << block->blockNb;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	// reported block number consistent with expectation
	if (block->blockNb != res_->iBlock_) {
		res_->nbErrors_++;
		stringstream oss;
		oss << "Incorrect block number." << " Expected:" << res_->iBlock_
				<< " Received:" << block->blockNb;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	// reported payload size consistent with expectation
	bufSize = bufRef->getDataSize();
	segSizeExpected = bufSize - frameSize - sizeof(frlh_t);
	segSize = frlHeader->segsize & FRL_SEGSIZE_MASK;
	if (segSize != segSizeExpected) {
		res_->nbErrors_++;
		stringstream oss;
		oss << "FRL header segment size is not as expected." << " Expected:"
				<< segSizeExpected << " Received:" << segSize;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	// Check that FU and FRL headers agree on end of super-fragment
	bool fuLastBlockInSuperFrag = (block->blockNb
			== (block->nbBlocksInSuperFragment - 1));
	bool frlLastBlockInSuperFrag = ((frlHeader->segsize & FRL_LAST_SEGM) != 0);
	if (fuLastBlockInSuperFrag != frlLastBlockInSuperFrag) {
		res_->nbErrors_++;
		stringstream oss;
		oss << "FU / FRL header end-of-superfragment mismatch."
				<< " FU header:" << fuLastBlockInSuperFrag << " FRL header:"
				<< frlLastBlockInSuperFrag;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	return;
}

//______________________________________________________________________________
void ResourceChecker::findFEDs() throw (evf::Exception) {
	UChar_t* superFragAddr = 0;
	UInt_t superFragSize = 0;

	UChar_t *fedTrailerAddr = 0;
	UChar_t *fedHeaderAddr = 0;

	UInt_t fedSize = 0;
	UInt_t sumOfFedSizes = 0;
	UInt_t evtNumber = 0;

	UShort_t crc = 0;
	UShort_t crcChk = 0;

	fedt_t *fedTrailer = 0;
	fedh_t *fedHeader = 0;

	superFragAddr = res_->shmCell_->superFragAddr(res_->iSuperFrag_);
	superFragSize = res_->shmCell_->superFragSize(res_->iSuperFrag_);
	fedTrailerAddr = superFragAddr + superFragSize - sizeof(fedt_t);

	while (fedTrailerAddr > superFragAddr) {

		fedTrailer = (fedt_t*) fedTrailerAddr;
		fedSize = (fedTrailer->eventsize & FED_EVSZ_MASK) << 3;
		sumOfFedSizes += fedSize;

		// check for fed trailer id
		if ((fedTrailer->eventsize & FED_TCTRLID_MASK) != FED_TCTRLID) {
			res_->nbErrors_++;
			stringstream oss;
			oss << "Missing FED trailer id." << " evtNumber:"
					<< res_->evtNumber_ << " iSuperFrag:" << res_->iSuperFrag_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		fedHeaderAddr = fedTrailerAddr - fedSize + sizeof(fedt_t);

		// check that fed header is within buffer
		if (fedHeaderAddr < superFragAddr) {
			res_->nbErrors_++;
			stringstream oss;
			oss << "FED header address out-of-bounds." << " evtNumber:"
					<< res_->evtNumber_ << " iSuperFrag:" << res_->iSuperFrag_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		// check that payload starts within buffer
		if ((fedHeaderAddr + sizeof(fedh_t)) > (superFragAddr + superFragSize)) {
			res_->nbErrors_++;
			stringstream oss;
			oss << "FED payload out-of-bounds." << " evtNumber:"
					<< res_->evtNumber_ << " iSuperFrag:" << res_->iSuperFrag_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		fedHeader = (fedh_t*) fedHeaderAddr;

		// check for fed header id
		if ((fedHeader->eventid & FED_HCTRLID_MASK) != FED_HCTRLID) {
			res_->nbErrors_++;
			stringstream oss;
			oss << "Missing FED header id." << " evtNumber:"
					<< res_->evtNumber_ << " iSuperFrag:" << res_->iSuperFrag_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		UInt_t fedId = (fedHeader->sourceid & REAL_SOID_MASK) >> 8;

		// check evtNumber consisency
		evtNumber = fedHeader->eventid & FED_LVL1_MASK;
		if (evtNumber != res_->evtNumber_) {
			res_->nbErrors_++;
			stringstream oss;
			oss << "FU / FED evtNumber mismatch." << " FU:" << res_->evtNumber_
					<< " FED:" << evtNumber << " fedid:" << fedId;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		// check that fedid is within valid ranges
		if (fedId >= 1024 || (res_->doFedIdCheck_ && (!FEDNumbering::inRange(
				fedId)))) {
			LOG4CPLUS_WARN(
					res_->log_,
					"Invalid fedid. Data will still be logged" << " evtNumber:"
							<< res_->evtNumber_ << " fedid:" << fedId);
			res_->nbErrors_++;
		}

		// check if a previous fed has already claimed same fed id

		if (res_->fedSize_[fedId] != 0) {
			LOG4CPLUS_ERROR(
					res_->log_,
					"Duplicated fedid. Data will be lost for" << " evtNumber:"
							<< res_->evtNumber_ << " fedid:" << fedId);
			res_->nbErrors_++;
		}

		if (fedId < 1024)
			res_->fedSize_[fedId] = fedSize;

		//if gtp EVM block is available set cell event number to global partition-independent trigger number
		//daq block partition-independent event number is left as an option in case of problems

		if (fedId == res_->gtpeId_)
			if (evf::evtn::gtpe_board_sense(fedHeaderAddr))
				res_->shmCell_->setEvtNumber(evf::evtn::gtpe_get(fedHeaderAddr));
		if (res_->useEvmBoard_ && (fedId == res_->gtpEvmId_))
			// UPDATED
			if (evf::evtn::evm_board_sense(fedHeaderAddr, fedSize)) {
				res_->shmCell_->setEvtNumber(
						evf::evtn::get(fedHeaderAddr, true));
				res_->shmCell_->setLumiSection(evf::evtn::getlbn(fedHeaderAddr));
			}
		if (!res_->useEvmBoard_ && (fedId == res_->gtpDaqId_))

			if (evf::evtn::daq_board_sense(fedHeaderAddr)) {
				res_->shmCell_->setEvtNumber(
						evf::evtn::get(fedHeaderAddr, false));
			}
		// crc check
		if (res_->doCrcCheck_) {
			UInt_t conscheck = fedTrailer->conscheck;
			crc = ((fedTrailer->conscheck & FED_CRCS_MASK) >> FED_CRCS_SHIFT);
			fedTrailer->conscheck &= (~FED_CRCS_MASK);
			fedTrailer->conscheck &= (~FED_RBIT_MASK);
			crcChk = compute_crc(fedHeaderAddr, fedSize);
			if (res_->nextEventWillHaveCRCError_ && random() > RAND_MAX / 2) {
				crc--;
				res_->nextEventWillHaveCRCError_ = false;
			}
			if (crc != crcChk) {
				ostringstream oss;
				oss << "crc check failed." << " evtNumber:" << res_->evtNumber_
						<< " fedid:" << fedId << " crc:" << crc << " chk:"
						<< crcChk;
				LOG4CPLUS_INFO(res_->log_, oss.str());
				XCEPT_DECLARE(evf::Exception, sentinelException, oss.str());
				res_->app_->notifyQualified("error", sentinelException);
				res_->nbErrors_++;
				res_->nbCrcErrors_++;
			}
			fedTrailer->conscheck = conscheck;
		}

		// mark fed
		if (!res_->shmCell_->markFed(fedId, fedSize, fedHeaderAddr)) {
			res_->nbErrors_++;
			stringstream oss;
			oss << "Failed to mark fed in buffer." << " evtNumber:"
					<< res_->evtNumber_ << " fedId:" << fedId << " fedSize:"
					<< fedSize << " fedAddr:0x" << hex
					<< (unsigned long) fedHeaderAddr << dec;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		// Move to the next fed trailer
		fedTrailerAddr = fedTrailerAddr - fedSize;
	}

	// check that we indeed end up on the starting address of the buffer
	if ((fedTrailerAddr + sizeof(fedh_t)) != superFragAddr) {
		stringstream oss;
		oss << "First FED in superfragment ouf-of-bound." << " evtNumber:"
				<< res_->evtNumber_ << " iSuperFrag:" << res_->iSuperFrag_;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	return;
}
