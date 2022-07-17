#!/usr/bin/perl

use lib "..";

use GenMul;

### If you're going to run GMtest.cxx and you do some changes here
### you *MUST* bring DIM, DOM and pattern assumptions in sync!

my $DIM = 3;
my $DOM = 6;

$a = new GenMul::MatrixSym('name'=>'a', 'M'=>$DIM);
$a->set_pattern(<<"FNORD");
x
x 1 
x x x
FNORD

$b = new GenMul::Matrix('name'=>'b', 'M'=>$DIM, 'N'=>$DOM);
$b->set_pattern(<<"FNORD");
x x x x 0 x
x 1 x 1 0 x
x x x x 0 x
FNORD

$c = new GenMul::Matrix('name'=>'c', 'M'=>$DIM, 'N'=>$DOM);


$bt = new GenMul::MatrixTranspose($b);
$bt->print_info();
$bt->print_pattern();

$ct = new GenMul::Matrix('name'=>'c', 'M'=>$DOM, 'N'=>$DIM);

# ----------------------------------------------------------------------

# E.g. to skip matrix size check:
#   $m = new GenMul::Multiply('no_size_check'=>1);
# Note that matrix dimensions that you pass into auto-generated
# function still has to match matrix dimensions set here.

$m = new GenMul::Multiply;

$m->dump_multiply_std_and_intrinsic("multify.ah", $a, $b, $c);

$m->dump_multiply_std_and_intrinsic("multify-transpose.ah", $bt, $a, $ct);

# To separate outputs of each function:
#
# open STD, ">multify.ah";
# select STD;

# $m->multiply_standard($a, $b, $c);

# close STD;

# # print "\n", '-' x 80, "\n\n";

# open INT, ">multify_intr.ah";
# select INT;

# $m->multiply_intrinsic($a, $b, $c);

# close INT;
