#include "Alignment/LaserAlignment/interface/LASGlobalDataLoop.h"

LASGlobalDataLoop::LASGlobalDataLoop(loop_type lp_tp) :
  the_loop_type(lp_tp),
  det(0),
  beam(0),
  ring(-1),
  zpos(0),
  loop_finished(false),
  max_det(3),
  max_ring(1),
  max_zpos(8)
{
  switch(the_loop_type){
  case ALL:
    max_det = 3;
    max_ring = 1;
    max_zpos = 4;
    break;
  case TEC:
    ring = 0;
    max_det = 1;
    max_ring = 1;
    max_zpos = 8;
    break;
  case TEC_PLUS:
    ring = 0;
    max_det = 0;
    max_ring = 1;
    max_zpos = 8;
    break;
  case TEC_MINUS:
    det = 1;
    ring = 0;
    max_det = 1;
    max_ring = 1;
    max_zpos = 8;
    break;
  case AT:
    det = 0;
    max_det = 3;
    max_ring = -1;
    max_zpos = 4;
    break;
  case TIB:
    det = 2;
    max_det = 2;
    max_ring = -1;
    max_zpos = 5;
    break;
  case TOB:
    det = 3;
    max_det = 3;
    max_ring = -1;
    max_zpos = 5;
    break;
  case TEC_AT:
    max_det = 1;
    max_ring = -1;
    max_zpos = 4;
    break;
  case TEC_PLUS_AT:
    max_det = 0;
    max_ring = -1;
    max_zpos = 4;
    break;
  case TEC_MINUS_AT:
    det = 1;
    max_det = 1;
    max_ring = -1;
    max_zpos = 4;
    break;
  default:
    throw(std::runtime_error("invalid loop type in LASGlobalDataLoop constructor"));
  }
}

void LASGlobalDataLoop::inspect(std::ostream & out)
{
  out << "det: " << det << " ring: " << ring << " beam: " << beam << " zpos: " << zpos << std::endl;
}

bool LASGlobalDataLoop::next()
{
  if(loop_finished)return false;

  if(zpos < max_zpos){
    zpos++;
    return true;
  }
  zpos = 0;
  if(beam < 7){
    beam ++;
    return true;
  }
  beam = 0;
  if(ring < max_ring){
    ring ++;
    if(ring >= 0) max_zpos = 8;
    return true;
  }

  if(det < max_det){
    det++;
    // reset ring
    ring = -1;
    if(det == 1 && the_loop_type == TEC) ring = 0;
    // reset max_zpos and max_ring
    if(det > 1){
      max_ring = -1;
      max_zpos = 5;
    }
    else if(ring == -1) max_zpos = 4;
    return true;
  }

    //  enum loop_type{ALL, TEC_PLUS, TEC_MINUS, TEC, AT, TIB, TOB, TEC_PLUS_AT, TEC_MINUS_AT, TEC_AT};

  loop_finished = true;
  return false;
}
