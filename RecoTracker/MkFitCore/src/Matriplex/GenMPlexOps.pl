#!/usr/bin/perl

use lib "../Matriplex";

use GenMul;
use warnings;

#------------------------------------------------------------------------------
### simple general 3x3 matrix times 3 vector multiplication for CF MPlex

$A = new GenMul::Matrix('name'=>'a', 'M'=>3, 'N'=>3);

$B = new GenMul::Matrix('name'=>'b', 'M'=>3, 'N'=>1);

$C = new GenMul::Matrix('name'=>'c', 'M'=>3, 'N'=>1);

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("CFMatrix33Vector3.ah",
                                    $A, $B, $C);

#------------------------------------------------------------------------------
###updateParametersMPlex -- propagated errors in CCS coordinates
# propErr_ccs = jac_ccs * propErr * jac_ccsT

$jac_ccs = new GenMul::Matrix('name'=>'a', 'M'=>6, 'N'=>6);
$jac_ccs->set_pattern(<<"FNORD");
1 0 0 0 0 0
0 1 0 0 0 0
0 0 1 0 0 0
0 0 0 x x 0
0 0 0 x x 0
0 0 0 x x x
FNORD

$propErr = new GenMul::MatrixSym('name'=>'b', 'M'=>6, 'N'=>6);

$temp   = new GenMul::Matrix('name'=>'c', 'M'=>6, 'N'=>6);

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("CCSErr.ah",
                                    $jac_ccs, $propErr, $temp);

$jac_ccsT = new GenMul::MatrixTranspose($jac_ccs);
$propErr_ccs = new GenMul::MatrixSym('name'=>'c', 'M'=>6, 'N'=>6);
$temp  ->{name} = 'b';

$m->dump_multiply_std_and_intrinsic("CCSErrTransp.ah",
                                    $temp, $jac_ccsT, $propErr_ccs);

#------------------------------------------------------------------------------
###updateParametersMPlex -- updated errors in cartesian coordinates
# outErr = jac_back_ccs * outErr_ccs * jac_back_ccsT

$jac_back_ccs = new GenMul::Matrix('name'=>'a', 'M'=>6, 'N'=>6);
$jac_back_ccs->set_pattern(<<"FNORD");
1 0 0 0 0 0
0 1 0 0 0 0
0 0 1 0 0 0
0 0 0 x x 0
0 0 0 x x 0
0 0 0 x 0 x
FNORD

$outErr_ccs = new GenMul::MatrixSym('name'=>'b', 'M'=>6, 'N'=>6);

$temp   = new GenMul::Matrix('name'=>'c', 'M'=>6, 'N'=>6);

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("CartesianErr.ah",
                                    $jac_back_ccs, $outErr_ccs, $temp);

$jac_back_ccsT = new GenMul::MatrixTranspose($jac_back_ccs);
$outErr = new GenMul::MatrixSym('name'=>'c', 'M'=>6, 'N'=>6);
$temp  ->{name} = 'b';

$m->dump_multiply_std_and_intrinsic("CartesianErrTransp.ah",
                                    $temp, $jac_back_ccsT, $outErr);

#------------------------------------------------------------------------------
###updateParametersMPlex -- first term to get kalman gain (H^T*G)
# temp = rot * resErr_loc

$rot = new GenMul::Matrix('name'=>'a', 'M'=>3, 'N'=>3);
$rot->set_pattern(<<"FNORD");
x 0 x
x 0 x
0 1 0
FNORD

$resErr_loc = new GenMul::MatrixSym('name'=>'b', 'M'=>3, 'N'=>3);

$temp   = new GenMul::Matrix('name'=>'c', 'M'=>3, 'N'=>3);

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("KalmanHTG.ah",
                                    $rot, $resErr_loc, $temp);


#------------------------------------------------------------------------------
###updateParametersMPlex -- kalman gain
# K = propErr_ccs * resErrTmpLH

$propErr_ccs = new GenMul::MatrixSym('name'=>'a', 'M'=>6, 'N'=>6);

$resErrTmpLH  = new GenMul::Matrix('name'=>'b', 'M'=>6, 'N'=>3);
$resErrTmpLH->set_pattern(<<"FNORD");
x x 0
x x 0
x x 0
0 0 0
0 0 0
0 0 0
FNORD

$K   = new GenMul::Matrix('name'=>'c', 'M'=>6, 'N'=>3);

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("KalmanGain.ah",
                                    $propErr_ccs, $resErrTmpLH, $K);
#------------------------------------------------------------------------------
###updateParametersMPlex -- kalman gain
# K = propErr * resErr2x2

$propErr = new GenMul::MatrixSym('name'=>'a', 'M'=>6, 'N'=>6);

$resErr2x2  = new GenMul::MatrixSym('name'=>'b', 'M'=>2, 'N'=>2);

$K   = new GenMul::Matrix('name'=>'c', 'M'=>6, 'N'=>2);

{
  my $m_kg = new GenMul::Multiply('no_size_check' => 1);

  $m_kg->dump_multiply_std_and_intrinsic("KalmanGain62.ah",
					 $propErr, $resErr2x2, $K);
}

#------------------------------------------------------------------------------
###updateParametersMPlex -- KH
# KH = K * H

$K   = new GenMul::Matrix('name'=>'a', 'M'=>6, 'N'=>3);
$K->set_pattern(<<"FNORD");
x x 0
x x 0
x x 0
x x 0
x x 0
x x 0
FNORD

$H   = new GenMul::Matrix('name'=>'b', 'M'=>3, 'N'=>6);
$H->set_pattern(<<"FNORD");
x x 0 0 0 0
0 0 1 0 0 0
x x 0 0 0 0
FNORD

$KH   = new GenMul::Matrix('name'=>'c', 'M'=>6, 'N'=>6);

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("KH.ah",
                                    $K, $H, $KH);

#------------------------------------------------------------------------------
###updateParametersMPlex -- KH * C
# temp = KH * propErr_ccs

$KH   = new GenMul::Matrix('name'=>'a', 'M'=>6, 'N'=>6);
$KH->set_pattern(<<"FNORD");
x x x 0 0 0
x x x 0 0 0
x x x 0 0 0
x x x 0 0 0
x x x 0 0 0
x x x 0 0 0
FNORD

$propErr_ccs = new GenMul::MatrixSym('name'=>'b', 'M'=>6, 'N'=>6);

$temp   = new GenMul::MatrixSym('name'=>'c', 'M'=>6, 'N'=>6);

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("KHC.ah",
                                    $KH, $propErr_ccs, $temp);

#------------------------------------------------------------------------------

###updateParametersMPlex -- KH * C with KH=K dim 6x2
# temp = KH * propErr

$KH   = new GenMul::Matrix('name'=>'a', 'M'=>6, 'N'=>2);
$KH->set_pattern(<<"FNORD");
x x
x x
x x
x x
x x
x x
FNORD

$propErr = new GenMul::MatrixSym('name'=>'b', 'M'=>6, 'N'=>6);

$temp   = new GenMul::MatrixSym('name'=>'c', 'M'=>6, 'N'=>6);

{
  my $m_kg = new GenMul::Multiply('no_size_check' => 1);

  $m_kg->dump_multiply_std_and_intrinsic("K62HC.ah",
				      $KH, $propErr, $temp);
}

#------------------------------------------------------------------------------

### computeChi2MPlex -- similarity to rotate errors, two ops.
# resErr_loc = rotT * resErr_glo * rotTT

$rotT = new GenMul::Matrix('name'=>'a', 'M'=>3, 'N'=>3);
$rotT->set_pattern(<<"FNORD");
x x 0
0 0 1
x x 0
FNORD

$resErr_glo = new GenMul::MatrixSym('name'=>'b', 'M'=>3, 'N'=>3);

$temp   = new GenMul::Matrix('name'=>'c', 'M'=>3, 'N'=>3);

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("ProjectResErr.ah",
                                    $rotT, $resErr_glo, $temp);

$roTT = new GenMul::MatrixTranspose($rotT);
$resErr_loc = new GenMul::MatrixSym('name'=>'c', 'M'=>3, 'N'=>3);
$temp  ->{name} = 'b';

$m->dump_multiply_std_and_intrinsic("ProjectResErrTransp.ah",
                                    $temp, $roTT, $resErr_loc);

#------------------------------------------------------------------------------

### Propagate Helix To R -- final similarity, two ops.

# outErr = errProp * outErr * errPropT
#   outErr is symmetric

my $DIM = 6;

$errProp = new GenMul::Matrix('name'=>'a', 'M'=>$DIM, 'N'=>$DIM);
$errProp->set_pattern(<<"FNORD");
x x 0 x x 0
x x 0 x x 0
x x 1 x x x
x x 0 x x 0
x x 0 x x 0
0 0 0 0 0 1
FNORD
#switch to the one below when moving to CCS coordinates only
#x x 0 x x 0
#x x 0 x x 0
#x x 1 x x x
#0 0 0 1 0 0
#x x 0 x x 0
#0 0 0 0 0 1
#FNORD

$outErr = new GenMul::MatrixSym('name'=>'b', 'M'=>$DIM, 'N'=>$DIM);

$temp   = new GenMul::Matrix('name'=>'c', 'M'=>$DIM, 'N'=>$DIM);


$errPropT = new GenMul::MatrixTranspose($errProp);
$errPropT->print_info();
$errPropT->print_pattern();

# ----------------------------------------------------------------------

$m = new GenMul::Multiply;

# outErr and c are just templates ...

$m->dump_multiply_std_and_intrinsic("MultHelixProp.ah",
                                    $errProp, $outErr, $temp);

$temp  ->{name} = 'b';
$outErr->{name} = 'c';

### XXX fix this ... in accordance with what is in Propagation.cc
$m->dump_multiply_std_and_intrinsic("MultHelixPropTransp.ah",
                                    $temp, $errPropT, $outErr);

#######################################
###          ENDCAP version         ###
#######################################

$errProp->set_pattern(<<"FNORD");
1 0 x x x x
0 1 x x x x
0 0 0 0 0 0
0 0 0 1 0 0
0 0 x x 1 x
0 0 0 0 0 1
FNORD

$temp  ->{name} = 'c';
$outErr->{name} = 'b';

$errPropT = new GenMul::MatrixTranspose($errProp);
$m->dump_multiply_std_and_intrinsic("MultHelixPropEndcap.ah",
                                    $errProp, $outErr, $temp);

$temp  ->{name} = 'b';
$outErr->{name} = 'c';

### XXX fix this ... in accordance with what is in Propagation.cc
$m->dump_multiply_std_and_intrinsic("MultHelixPropTranspEndcap.ah",
                                    $temp, $errPropT, $outErr);


##############################
### updateParameters       ###
##############################

#declared first on its own because propErr sees many uses
my $propErr_M = 6;
$propErr = new GenMul::MatrixSym('name' => 'a',
                                 'M'    => $propErr_M); #will have to remember to re'name' it based on location in function

my $propErrT_M = 6;
$propErrT = new GenMul::MatrixTranspose($propErr); #will have to remember to re'name' it based on location in function



### kalmanGain =  = propErr * (projMatrixT * resErrInv)
$resErrInv = new GenMul::MatrixSym('name'=>'b', 'M'=>3, 'N'=>3);

$kalmanGain = new GenMul::Matrix('name'=>'c', 'M' => 6, 'N' => 3);

{
  my $m_kg = new GenMul::Multiply('no_size_check' => 1);

  $m_kg->dump_multiply_std_and_intrinsic("upParam_MultKalmanGain.ah",
                                         $propErr, $resErrInv, $kalmanGain);
}


### updatedErrs = propErr - propErr^T * simil * propErr
# Going to skip the subtraction for now
my $simil_M = 6;
$simil = new GenMul::MatrixSym('name'=>'a', 'M'=>$simil_M);
$simil->set_pattern(<<"FNORD");
x
x x
x x x
0 0 0 0
0 0 0 0 0
0 0 0 0 0 0
FNORD

$propErr->{name} = 'b';

my $temp_simil_x_propErr_M = 6;
my $temp_simil_x_propErr_N = 6;
$temp_simil_x_propErr = new GenMul::Matrix('name'=>'c',
                                           'M'=>$temp_simil_x_propErr_M,
                                           'N'=>$temp_simil_x_propErr_N);

$m->dump_multiply_std_and_intrinsic("upParam_simil_x_propErr.ah",
                                    $simil, $propErr, $temp_simil_x_propErr);

$temp_simil_x_propErr->{name} = 'b';									 
$temp_simil_x_propErr->set_pattern(<<"FNORD");
x x x x x x
x x x x x x
x x x x x x
0 0 0 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0
FNORD

#? This one is symmetric but the output can't handle it... need to fix
#$temp_propErrT_x_simil_propErr = new GenMul::MatrixSym('name'=>'c', 'M'=>$propErrT_M, 'N'=>$temp_simil_x_propErr_N);


$temp_propErrT_x_simil_propErr = new GenMul::MatrixSym('name'=>'c', 'M'=>$propErrT_M);

$m->dump_multiply_std_and_intrinsic("upParam_propErrT_x_simil_propErr.ah",
                                    $propErrT, $temp_simil_x_propErr, $temp_propErrT_x_simil_propErr);
									

{
  my $temp = new GenMul::MatrixSym('name' => 'c', 'M' => 6);

  my $m_kg = new GenMul::Multiply('no_size_check' => 1);

  $kalmanGain->{name} = 'a';

  $m_kg->dump_multiply_std_and_intrinsic("upParam_kalmanGain_x_propErr.ah",
                                         $kalmanGain, $propErr, $temp);
}
