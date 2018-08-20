#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"
#include "HepMC/SimpleVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

static int skipWhitespace(std::istream &in)
{
	int ch;
	do {
		ch = in.get();
	} while(std::isspace(ch));
	if (ch != std::istream::traits_type::eof())
		in.putback(ch);
	return ch;
}

namespace lhef {

LHEEvent::LHEEvent(const boost::shared_ptr<LHERunInfo> &runInfo,
                   std::istream &in) :
  runInfo(runInfo), weights_(0), counted(false), 
  readAttemptCounter(0), npLO_(-99), npNLO_(-99)
  
{
	hepeup.NUP = 0;
	hepeup.XPDWUP.first = hepeup.XPDWUP.second = 0.0;

	in >> hepeup.NUP >> hepeup.IDPRUP >> hepeup.XWGTUP
	   >> hepeup.SCALUP >> hepeup.AQEDUP >> hepeup.AQCDUP;
	if (!in.good())
		throw cms::Exception("InvalidFormat")
			<< "Les Houches file contained invalid"
			   " event header." << std::endl;

	// store the original value of XWGTUP for the user
	originalXWGTUP_ = hepeup.XWGTUP;

	int idwtup = runInfo->getHEPRUP()->IDWTUP;
	if (idwtup >= 0 && hepeup.XWGTUP < 0) {
		edm::LogWarning("Generator|LHEInterface")
			<< "Non-allowed negative event weight encountered."
			<< std::endl;
		hepeup.XWGTUP = std::abs(hepeup.XWGTUP);
	}

	if (std::abs(idwtup) == 3 && std::abs(hepeup.XWGTUP) != 1.) {
		edm::LogInfo("Generator|LHEInterface")
			<< "Event weight not set to one for abs(IDWTUP) == 3"
			<< std::endl;
		hepeup.XWGTUP = hepeup.XWGTUP > 0. ? 1.0 : -1.0;
	}

	hepeup.resize();

	for(int i = 0; i < hepeup.NUP; i++) {
		in >> hepeup.IDUP[i] >> hepeup.ISTUP[i]
		   >> hepeup.MOTHUP[i].first >> hepeup.MOTHUP[i].second
		   >> hepeup.ICOLUP[i].first >> hepeup.ICOLUP[i].second
		   >> hepeup.PUP[i][0] >> hepeup.PUP[i][1] >> hepeup.PUP[i][2]
		   >> hepeup.PUP[i][3] >> hepeup.PUP[i][4]
		   >> hepeup.VTIMUP[i] >> hepeup.SPINUP[i];
		if (!in.good())
			throw cms::Exception("InvalidFormat")
				<< "Les Houches file contained invalid event"
				   " in particle line " << (i + 1)
				<< "." << std::endl;
	}

	while(skipWhitespace(in) == '#') {
	  std::string line;
	  std::getline(in, line);
	  std::istringstream ss(line);
	  std::string tag;
	  ss >> tag;
	  if (tag == "#pdf") {
	    pdf.reset(new PDF);
	    ss >> pdf->id.first >> pdf->id.second
	       >> pdf->x.first >> pdf->x.second
	       >> pdf->scalePDF
	       >> pdf->xPDF.first >> pdf->xPDF.second;
	    if (ss.bad()) {
	      edm::LogWarning("Generator|LHEInterface")
		<< "Les Houches event contained"
		" unparseable PDF information."
					<< std::endl;
	      pdf.reset();
	    } else
	      continue;
	  }	  
	  comments.push_back(line + "\n");
	}
	
	if (!in.eof())
	  edm::LogWarning("Generator|LHEInterface")
	    << "Les Houches file contained spurious"
	    " content after event data." << std::endl;
}

LHEEvent::LHEEvent(const boost::shared_ptr<LHERunInfo> &runInfo,
                   const HEPEUP &hepeup) :
	runInfo(runInfo), hepeup(hepeup), counted(false), readAttemptCounter(0),
        npLO_(-99), npNLO_(-99)
{
}

LHEEvent::LHEEvent(const boost::shared_ptr<LHERunInfo> &runInfo,
                   const HEPEUP &hepeup,
                   const LHEEventProduct::PDF *pdf,
                   const std::vector<std::string> &comments) :
	runInfo(runInfo), hepeup(hepeup), pdf(pdf ? new PDF(*pdf) : nullptr),
	comments(comments), counted(false), readAttemptCounter(0),
	npLO_(-99), npNLO_(-99)
{
}

LHEEvent::LHEEvent(const boost::shared_ptr<LHERunInfo> &runInfo,
                   const LHEEventProduct &product) :
	runInfo(runInfo), hepeup(product.hepeup()),
	pdf(product.pdf() ? new PDF(*product.pdf()) : nullptr),
	weights_(product.weights()),
	comments(product.comments_begin(), product.comments_end()),
	counted(false), readAttemptCounter(0),
	originalXWGTUP_(product.originalXWGTUP()), scales_(product.scales()),
	npLO_(product.npLO()), npNLO_(product.npNLO())
{
}

LHEEvent::~LHEEvent()
{
}

void LHEEvent::removeParticle(HEPEUP &hepeup, int index)
{
	index--;
	if (index < 0 || index >= hepeup.NUP) {
		edm::LogError("Generator|LHEInterface")
			<< "removeParticle: Index " << (index + 1)
			<< " out of bounds." << std::endl;
		return;
	}

	std::pair<int, int> mo = hepeup.MOTHUP[index];

	hepeup.IDUP.erase(hepeup.IDUP.begin() + index);
	hepeup.ISTUP.erase(hepeup.ISTUP.begin() + index);
	hepeup.MOTHUP.erase(hepeup.MOTHUP.begin() + index);
	hepeup.ICOLUP.erase(hepeup.ICOLUP.begin() + index);
	hepeup.PUP.erase(hepeup.PUP.begin() + index);
	hepeup.VTIMUP.erase(hepeup.VTIMUP.begin() + index);
	hepeup.SPINUP.erase(hepeup.SPINUP.begin() + index);

	hepeup.NUP--;

	index++;
	for(int i = 0; i < hepeup.NUP; i++) {
		if (hepeup.MOTHUP[i].first == index) {
			if (hepeup.MOTHUP[i].second > 0)
				edm::LogError("Generator|LHEInterface")
					<< "removeParticle: Particle "
					<< (i + 2) << " has two mothers."
					<< std::endl;
			hepeup.MOTHUP[i] = mo;
		}

		if (hepeup.MOTHUP[i].first > index)
			hepeup.MOTHUP[i].first--;
		if (hepeup.MOTHUP[i].second > index)
			hepeup.MOTHUP[i].second--;
	}
}

void LHEEvent::removeResonances(const std::vector<int> &ids)
{
	for(int i = 1; i <= hepeup.NUP; i++) {
		int id = std::abs(hepeup.IDUP[i - 1]);
		if (hepeup.MOTHUP[i - 1].first > 0 &&
		    hepeup.MOTHUP[i - 1].second > 0 &&
		    std::find(ids.begin(), ids.end(), id) != ids.end())
			removeParticle(hepeup, i--);
	}
}

void LHEEvent::count(LHERunInfo::CountMode mode,
                     double weight, double matchWeight)
{
	if (counted)
		edm::LogWarning("Generator|LHEInterface")
			<< "LHEEvent::count called twice on same event!"
			<< std::endl;

	runInfo->count(hepeup.IDPRUP, mode, hepeup.XWGTUP,
	               weight, matchWeight);

	counted = true;
}

void LHEEvent::fillPdfInfo(HepMC::PdfInfo *info) const
{
	if (pdf.get()) {
		info->set_id1(pdf->id.first);
		info->set_id2(pdf->id.second);
		info->set_x1(pdf->x.first);
		info->set_x2(pdf->x.second);
		info->set_pdf1(pdf->xPDF.first);
		info->set_pdf2(pdf->xPDF.second);
		info->set_scalePDF(pdf->scalePDF);
	} else if (hepeup.NUP >= 2) {
		const HEPRUP *heprup = getHEPRUP();
		info->set_id1(hepeup.IDUP[0] == 21 ? 0 : hepeup.IDUP[0]);
		info->set_id2(hepeup.IDUP[1] == 21 ? 0 : hepeup.IDUP[1]);
		info->set_x1(std::abs(hepeup.PUP[0][2] / heprup->EBMUP.first));
		info->set_x2(std::abs(hepeup.PUP[1][2] / heprup->EBMUP.second));
		info->set_pdf1(-1.0);
		info->set_pdf2(-1.0);
		info->set_scalePDF(hepeup.SCALUP);
	} else {
		info->set_x1(-1.0);
		info->set_x2(-1.0);
		info->set_pdf1(-1.0);
		info->set_pdf2(-1.0);
		info->set_scalePDF(hepeup.SCALUP);
	}
}

void LHEEvent::fillEventInfo(HepMC::GenEvent *event) const
{
	event->set_signal_process_id(hepeup.IDPRUP);
	event->set_event_scale(hepeup.SCALUP);
	event->set_alphaQED(hepeup.AQEDUP);
	event->set_alphaQCD(hepeup.AQCDUP);
}

std::auto_ptr<HepMC::GenEvent> LHEEvent::asHepMCEvent() const
{
	std::auto_ptr<HepMC::GenEvent> hepmc(new HepMC::GenEvent);

	hepmc->set_signal_process_id(hepeup.IDPRUP);
	hepmc->set_event_scale(hepeup.SCALUP);
	hepmc->set_alphaQED(hepeup.AQEDUP);
	hepmc->set_alphaQCD(hepeup.AQCDUP);

	unsigned int nup = hepeup.NUP; // particles in event

	// any particles in HEPEUP block?
	if (!nup) {
		edm::LogWarning("Generator|LHEInterface")
			<< "Les Houches Event does not contain any partons. "
			<< "Not much to convert." ;
		return hepmc;
	}

	// stores (pointers to) converted particles
	std::vector<HepMC::GenParticle*> genParticles;
	std::vector<HepMC::GenVertex*> genVertices;

	// I. convert particles
	for(unsigned int i = 0; i < nup; i++)
		genParticles.push_back(makeHepMCParticle(i));

	// II. loop again to build vertices
	for(unsigned int i = 0; i < nup; i++) {
		unsigned int mother1 = hepeup.MOTHUP.at(i).first;
		unsigned int mother2 = hepeup.MOTHUP.at(i).second;
		double cTau = hepeup.VTIMUP.at(i);	// decay time
	
		// current particle has a mother? --- Sorry, parent! We're PC.
		if (mother1) {
			mother1--;      // FORTRAN notation!
			if (mother2)
				mother2--;
			else
				mother2 = mother1;

			HepMC::GenParticle *in_par = genParticles.at(mother1);
			HepMC::GenVertex *current_vtx = in_par->end_vertex();  // vertex of first mother

			if (!current_vtx) {
				current_vtx = new HepMC::GenVertex(
						HepMC::FourVector(0, 0, 0, cTau));

				// add vertex to event
				genVertices.push_back(current_vtx);
			}

			for(unsigned int j = mother1; j <= mother2; j++) // set mother-daughter relations
				if (!genParticles.at(j)->end_vertex())
					current_vtx->add_particle_in(genParticles.at(j));

			// connect THIS outgoing particle to current vertex
			current_vtx->add_particle_out(genParticles.at(i));
		}
	}

	checkHepMCTree(hepmc.get());

	// III. restore color flow
	// ok, nobody knows how to do it so far...

	// IV. fill run information
	const HEPRUP *heprup = getHEPRUP();

	// set beam particles
	HepMC::GenParticle *b1 = new HepMC::GenParticle(
			HepMC::FourVector(0.0, 0.0, +heprup->EBMUP.first,
			                             heprup->EBMUP.first),
			heprup->IDBMUP.first);
	HepMC::GenParticle *b2 = new HepMC::GenParticle(
			HepMC::FourVector(0.0, 0.0, -heprup->EBMUP.second,
			                             heprup->EBMUP.second),
			heprup->IDBMUP.second);
	b1->set_status(3);
	b2->set_status(3);

	HepMC::GenVertex *v1 = new HepMC::GenVertex();
	HepMC::GenVertex *v2 = new HepMC::GenVertex();
	v1->add_particle_in(b1);
	v2->add_particle_in(b2);

	hepmc->add_vertex(v1);
	hepmc->add_vertex(v2);
	hepmc->set_beam_particles(b1, b2);

	// first two particles have to be the hard partons going into the interaction
	if (genParticles.size() >= 2) {
		if (!genParticles.at(0)->production_vertex() &&
		    !genParticles.at(1)->production_vertex()) {
			v1->add_particle_out(genParticles.at(0));
			v2->add_particle_out(genParticles.at(1));
		} else
			edm::LogWarning("Generator|LHEInterface")
				<< "Initial partons do already have a"
				   " production vertex. " << std::endl
				<< "Beam particles not connected.";
	} else
		edm::LogWarning("Generator|LHEInterface")
			<< "Can't find any initial partons to be"
			   " connected to the beam particles.";

	for(std::vector<HepMC::GenVertex*>::const_iterator iter = genVertices.begin();
	    iter != genVertices.end(); ++iter)
		hepmc->add_vertex(*iter);

	// do some more consistency checks
	for(unsigned int i = 0; i < nup; i++) {
		if (!genParticles.at(i)->parent_event()) {
			edm::LogWarning("Generator|LHEInterface")
				<< "Not all LHE particles could be stored"
				   "  stored in the HepMC event. "
				<< std::endl
				<< "Check the mother-daughter relations"
				   " in the given LHE input file.";
			break;
		}
	}

	hepmc->set_signal_process_vertex(
			const_cast<HepMC::GenVertex*>(
				findSignalVertex(hepmc.get(), false)));

	return hepmc;
}

HepMC::GenParticle *LHEEvent::makeHepMCParticle(unsigned int i) const
{
	HepMC::GenParticle *particle = new HepMC::GenParticle(
			HepMC::FourVector(hepeup.PUP.at(i)[0],
			                  hepeup.PUP.at(i)[1],
			                  hepeup.PUP.at(i)[2],
			                  hepeup.PUP.at(i)[3]),
			hepeup.IDUP.at(i));

	int status = hepeup.ISTUP.at(i);

	particle->set_generated_mass(hepeup.PUP.at(i)[4]);
	particle->set_status(status > 0 ? (status == 2 ? 3 : status) : 3);

	return particle;
}

bool LHEEvent::checkHepMCTree(const HepMC::GenEvent *event)
{
	double px = 0, py = 0, pz = 0, E = 0;

	for(HepMC::GenEvent::particle_const_iterator iter = event->particles_begin();
	    iter != event->particles_end(); iter++) {
		int status = (*iter)->status();
		HepMC::FourVector fv = (*iter)->momentum();

		// incoming particles
		if (status == 3 &&
		    *iter != event->beam_particles().first &&
		    *iter != event->beam_particles().second) {
			px -= fv.px();
			py -= fv.py();
			pz -= fv.pz();
			E  -= fv.e();
		}

		// outgoing particles
		if (status == 1) {
			px += fv.px();
			py += fv.py();
			pz += fv.pz();
			E  += fv.e();
		}
	}

	if (px*px + py*py + pz*pz + E*E > 0.1) {
		edm::LogWarning("Generator|LHEInterface")
			<< "Energy-momentum badly conserved. "
			<< std::setprecision(3)
			<< "sum p_i  = ["
			<< std::setw(7) << E << ", "
			<< std::setw(7) << px << ", "
			<< std::setw(7) << py << ", "
			<< std::setw(7) << pz << "]";

		return false;
	}

	return true;
}

const HepMC::GenVertex *LHEEvent::findSignalVertex(
				const HepMC::GenEvent *event, bool status3)
{
	double largestMass2 = -9.0e-30;
	const HepMC::GenVertex *vertex = nullptr;
	for(HepMC::GenEvent::vertex_const_iterator iter = event->vertices_begin();
	    iter != event->vertices_end(); ++iter) {
		if ((*iter)->particles_in_size() < 2)
			continue;
		if ((*iter)->particles_out_size() < 1 ||
		    ((*iter)->particles_out_size() == 1 &&
		     (!(*(*iter)->particles_out_const_begin())->end_vertex() ||
		      !(*(*iter)->particles_out_const_begin())
				->end_vertex()->particles_out_size())))
			continue;

		double px = 0.0, py = 0.0, pz = 0.0, E = 0.0;
		bool hadStatus3 = false;
		for(HepMC::GenVertex::particles_out_const_iterator iter2 =
					(*iter)->particles_out_const_begin();
		    iter2 != (*iter)->particles_out_const_end(); ++iter2) {
			hadStatus3 = hadStatus3 || (*iter2)->status() == 3;
			px += (*iter2)->momentum().px();
			py += (*iter2)->momentum().py();
			pz += (*iter2)->momentum().pz();
			E += (*iter2)->momentum().e();
		}
		if (status3 && !hadStatus3)
			continue;

		double mass2 = E * E - (px * px + py * py + pz * pz);
		if (mass2 > largestMass2) {
			vertex = *iter;
			largestMass2 = mass2;
		}
	}

	return vertex;
}

static void fixSubTree(HepMC::GenVertex *vertex,
                       HepMC::FourVector& _time,
                       std::set<const HepMC::GenVertex*> &visited)
{
	HepMC::FourVector time = _time;
	HepMC::FourVector curTime = vertex->position();
	bool needsFixup = curTime.t() < time.t();

	if (!visited.insert(vertex).second && !needsFixup)
		return;

	if (needsFixup)
		vertex->set_position(time);
	else
		time = curTime;

	for(HepMC::GenVertex::particles_out_const_iterator iter =
					vertex->particles_out_const_begin();
	    iter != vertex->particles_out_const_end(); ++iter) {
		HepMC::GenVertex *endVertex = (*iter)->end_vertex();
		if (endVertex)
			fixSubTree(endVertex, time, visited);
	}
}

void LHEEvent::fixHepMCEventTimeOrdering(HepMC::GenEvent *event)
{
	std::set<const HepMC::GenVertex*> visited;
	HepMC::FourVector zeroTime;
	for(HepMC::GenEvent::vertex_iterator iter = event->vertices_begin();
	    iter != event->vertices_end(); ++iter)
		fixSubTree(*iter, zeroTime, visited);
}

} // namespace lhef
