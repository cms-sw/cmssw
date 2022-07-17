########################################################################
########################################################################
# Top level package
########################################################################
########################################################################

package GenMul;

my $G_vec_width = 1;

########################################################################
########################################################################
# MATRIX CLASSES
########################################################################
########################################################################

########################################################################
# MBase -- matrix base class
########################################################################

package GenMul::MBase;

use Carp;

# Required arguments:
# - M
# - N (for non-symmetric matrices)
# - name: name of array
#
# Created members:
# - class
#
#
# Input matrix pattern can be set via function set_pattern(). The argument is
# a white-space separated string of x, 0, 1, describing the matrix elements.
# For symmetric matrices lower-left triangle must be given.
# Support for -1 could be added (but isn't trivial (unless unary - changes 
# the preceeding addition into subtraction; also this is a tough call
# for intrinsics)).
#
# Pattern could also be set for output matrix but is currently not supported.

sub new
{
  my $proto = shift;
  my $class = ref($proto) || $proto;
  my $S = {@_};
  bless($S, $class);

  # M, N checked in concrete classes

  croak "name must be set" unless defined $S->{name};

  $S->{class} = $class;

  return $S;
}

sub mat_size
{
  die "max_size() should be overriden in concrete matrix class";
}

sub idx
{
  die "idx() should be overriden in concrete matrix class";
}

sub row_col_in_range
{
  my ($S, $i, $j) = @_;

  return $i >= 0 and $i < $S->{M} and $j >= 0 and $j < $S->{N};
}

sub set_pattern
{
  my ($S, $pstr) = @_;

  @{$S->{pattern}} = split /\s+/, $pstr;

  croak "set_pattern number of entries does not match matrix size"
      unless scalar @{$S->{pattern}} == $S->mat_size();

  croak "set_pattern() input string contains invalid entry"
      if grep {$_ !~ /0|1|x/} @{$S->{pattern}};
}

sub pattern
{
  my ($S, $idx) = @_;

  die "pattern called with bad index."
      unless $idx >=0 and $idx < $S->mat_size();

  return defined $S->{pattern} ? $S->{pattern}[$idx] : 'x';
}

sub reg_name
{
  my ($S, $idx) = @_;

  die "reg_name called with bad index."
      unless $idx >=0 and $idx < $S->mat_size();

  return "$S->{name}_${idx}";
}

sub print_info
{
  my ($S) = @_;

  print "Class='$S->{class}', M=$S->{M}, N=$S->{N}, name='$S->{name}'\n";
}

sub print_pattern
{
  my ($S) = @_;

  for (my $i = 0; $i < $S->{M}; ++$i)
  {
    for (my $j = 0; $j < $S->{N}; ++$j)
    {
      print $S->pattern($S->idx($i, $j)), " ";
    }
    print "\n";
  }
}

########################################################################
# Matrix -- standard MxN matrix
########################################################################

package GenMul::Matrix; @ISA = ('GenMul::MBase');

use Carp;

sub new
{
  my $proto = shift;
  my $S = $proto->SUPER::new(@_);

  croak "M not set for $S->{class}" unless defined $S->{M};

  croak "N not set for $S->{class}" unless defined $S->{N};

  return $S;
}

sub mat_size
{
  my ($S) = @_;

  return $S->{M} * $S->{N};
}

sub idx
{
  my ($S, $i, $j) = @_;

  confess "$S->{class}::idx() i out of range"
      if $i < 0 or $i >= $S->{M};

  confess "$S->{class}::idx() j out of range"
      if $j < 0 or $j >= $S->{N};

  return $i * $S->{N} + $j;
}

########################################################################
# MatrixSym -- symmetric square matrix
########################################################################

package GenMul::MatrixSym; @ISA = ('GenMul::MBase');

use Carp;

# Offsets converting from full matrix indices to symmetric ones:
my @Offs;
@Offs[2] = [ 0, 1, 1, 2 ];
@Offs[3] = [ 0, 1, 3, 1, 2, 4, 3, 4, 5 ];
@Offs[4] = [ 0, 1, 3, 6, 1, 2, 4, 7, 3, 4, 5, 8, 6, 7, 8, 9 ];
@Offs[5] = [ 0, 1, 3, 6, 10, 1, 2, 4, 7, 11, 3, 4, 5, 8, 12, 6, 7, 8, 9, 13, 10, 11, 12, 13, 14 ];
@Offs[6] = [ 0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20 ];

sub new
{
  my $proto = shift;
  my $S = $proto->SUPER::new(@_);

  croak "M not set for $S->{class}" unless defined $S->{M};

  croak "N should not be set or should be equal to M for $S->{class}"
      if defined $S->{N} and $S->{N} != $S->{M};

  die "Offset array not defined for this dimension"
      unless defined @Offs[$S->{M}];

  die "Offset array of wrong dimension"
      unless scalar @{$Offs[$S->{M}]} == $S->{M} * $S->{M};

  $S->{N} = $S->{M} unless defined $S->{N};

  return $S;
}

sub mat_size
{
  my ($S) = @_;

  return ($S->{M} + 1) * $S->{M} / 2;
}

sub idx
{
  my ($S, $i, $j) = @_;

  confess "$S->{class}::idx() i out of range"
      if $i < 0 or $i >= $S->{M};

  confess "$S->{class}::idx() j out of range"
      if $j < 0 or $j >= $S->{N};

  return $Offs[$S->{M}][$i * $S->{N} + $j];
}


########################################################################
# MatrixTranspose -- wrapper for transpose of a matrix
########################################################################

package GenMul::MatrixTranspose; @ISA = ('GenMul::MBase');

use Carp;
use Scalar::Util 'blessed';


sub new
{
  my $proto = shift;
  my $mat   = shift;

  croak "Argument for $S->{class} is not a GenMul::MBase"
      unless blessed $mat and $mat->isa("GenMul::MBase");

  my $S = $proto->SUPER::new(@_, 'name'=>$mat->{name});


  $S->{matrix} = $mat;

  # Hack around dimensions -- these are accessed directly, everything
  # else goes through methods.

  $S->{M} = $S->{matrix}{N};
  $S->{N} = $S->{matrix}{M};

  return $S;
}

sub mat_size
{
  my ($S) = @_;

  return $S->{matrix}->mat_size();
}

sub idx
{
  my ($S, $i, $j) = @_;

  return $S->{matrix}->idx($j, $i);
}

sub pattern
{
  my ($S, $idx) = @_;

  return $S->{matrix}->pattern($idx);
}

sub print_info
{
  my ($S) = @_;

  print "Transpose of ";
  $S->{matrix}->print_info();
  print "    ";
  $S->SUPER::print_info();
}


########################################################################
########################################################################
# CODE GENERATION CLASSES
########################################################################
########################################################################

package GenMul::Multiply;

use Carp;
use Scalar::Util 'blessed';

use warnings;

# Optional arguments:
# - no_size_check: elements out of range are assumed to be 0

sub new
{
  my $proto = shift;
  my $class = ref($proto) || $proto;
  my $S = {@_};
  bless($S, $class);

  $S->{prefix}  = "      "    unless defined $S->{prefix};
  $S->{vectype} = "IntrVec_t" unless defined $S->{vectype};

  $S->{class} = $class;

  return $S;
}

sub check_multiply_arguments
{
  my ($S, $a, $b, $c) = @_;

  croak "Input a is not a GenMul::MBase"
      unless blessed $a and $a->isa("GenMul::MBase");

  croak "Input b is not a GenMul::MBase"
      unless blessed $b and $b->isa("GenMul::MBase");

  croak "Input c is not a GenMul::MBase"
      unless blessed $c and $c->isa("GenMul::MBase");

  unless ($S->{no_size_check})
  {
    croak "Input matrices a and b not compatible"
        unless $a->{N} == $b->{M};

    croak "Result matrix c of wrong dimensions"
        unless $c->{M} == $a->{M} and $c->{N} == $b->{N};
  }
  else
  {
    carp "Input matrices a and b not compatible -- running with no_size_check"
        unless $a->{N} == $b->{M};

    carp "Result matrix c of wrong dimensions -- running with no_size_check"
        unless $c->{M} == $a->{M} and $c->{N} == $b->{N};
  }

  croak "Result matrix c should not be a transpose (or check & implement this case in GenMul code)"
      if $c->isa("GenMul::MatrixTranspose");

  croak "Result matrix c has a pattern defined, this is not yet supported (but shouldn't be too hard)."
      if defined $c->{pattern};

  carp "Result matrix c is symmetric, GenMul hopes you know what you're doing"
      if $c->isa("GenMul::MatrixSym");

  $S->{a}{mat} = $a;
  $S->{b}{mat} = $b;
}

sub push_out
{
  my $S = shift;

  push @{$S->{out}}, join "", @_;
}

sub unshift_out
{
  my $S = shift;

  unshift @{$S->{out}}, join "", @_;
}

sub handle_all_zeros_ones
{
  my ($S, $zeros, $ones) = @_;

  if ($zeros or $ones)
  {
    my @zo;

    push @zo, "#ifdef AVX512_INTRINSICS";

    push @zo, "$S->{vectype} all_zeros = { " . join(", ", (0) x 16) . " };"
        if $zeros;

    push @zo, "$S->{vectype} all_ones  = { " . join(", ", (1) x 16) . " };"
        if $ones;

    push @zo, "#else";

    push @zo, "$S->{vectype} all_zeros = { " . join(", ", (0) x 8) . " };"
        if $zeros;

    push @zo, "$S->{vectype} all_ones  = { " . join(", ", (1) x 8) . " };"
        if $ones;

    push @zo, "#endif";

    push @zo, "";

    for $zol (reverse @zo)
    {
      $S->unshift_out($zol);
    }
  }
}

sub delete_temporaries
{
  my ($S) = @_;

  for $k ('idx', 'pat')
  {
    delete $S->{a};
    delete $S->{b};
  }
}

sub delete_loop_temporaries
{
  my ($S) = @_;

  for $k ('idx', 'pat')
  {
    delete $S->{a}{$k};
    delete $S->{b}{$k};
  }
}

sub generate_index_and_pattern
{
  my ($S, $x, $i1, $i2) = @_;

  if ($S->{no_size_check} and not $S->{$x}{mat}->row_col_in_range($i, $k))
  {
    $S->{$x}{pat} = '0';
  }
  else
  {
    $S->{$x}{idx} = $S->{$x}{mat}->idx($i1, $i2);
    $S->{$x}{pat} = $S->{$x}{mat}->pattern ($S->{$x}{idx});
  }
}

sub generate_indices_and_patterns_for_multiplication
{
  # Provide idcs and patterns for given indices

  my ($S, $i, $j, $k) = @_;

  $S->delete_loop_temporaries();

  $S->generate_index_and_pattern('a', $i, $k);
  $S->generate_index_and_pattern('b', $k, $j);
}

# ----------------------------------------------------------------------

sub generate_addend_standard
{
  my ($S, $x, $y) = @_;

  return undef if $S->{$x}{pat} eq '0' or  $S->{$y}{pat} eq '0';
  return "1"   if $S->{$x}{pat} eq '1' and $S->{$y}{pat} eq '1';

  my $xstr = sprintf "$S->{$x}{mat}{name}\[%2d*N+n]", $S->{$x}{idx};
  my $ystr = sprintf "$S->{$y}{mat}{name}\[%2d*N+n]", $S->{$y}{idx};

  return $xstr if $S->{$y}{pat} eq '1';
  return $ystr if $S->{$x}{pat} eq '1';

  return "${xstr}*${ystr}";
}

sub multiply_standard
{
  # Standard mutiplication - outputs unrolled C code, one line
  # per target matrix element.
  # Arguments: a, b, c   -- all GenMul::MBase with right dimensions.
  # Does:      c = a * b

  check_multiply_arguments(@_);

  my ($S, $a, $b, $c) = @_;

  my $is_c_symmetric = $c->isa("GenMul::MatrixSym");

  # With no_size_check matrices do not have to be compatible.
  my $k_max = $a->{N} <= $b->{M} ? $a->{N} : $b->{M};

  for (my $i = 0; $i < $c->{M}; ++$i)
  {
    my $j_max = $is_c_symmetric ?  $i + 1 : $c->{N};

    for (my $j = 0; $j < $j_max; ++$j)
    {
      my $x = $c->idx($i, $j);

      printf "$S->{prefix}$c->{name}\[%2d*N+n\] = ", $x;

      my @sum;

      for (my $k = 0; $k < $k_max; ++$k)
      {
        $S->generate_indices_and_patterns_for_multiplication($i, $j, $k);

        my $addend = $S->generate_addend_standard('a', 'b');

        push @sum, $addend if defined $addend;
      }
      if (@sum)
      {
        print join(" + ", @sum), ";";
      }
      else
      {
        print "0;"
      }
      print "\n";
    }
  }

  $S->delete_temporaries();
}

# ----------------------------------------------------------------------

sub generate_addend_gpu
{
  my ($S, $x, $y) = @_;

  return undef if $S->{$x}{pat} eq '0' or  $S->{$y}{pat} eq '0';
  return "1"   if $S->{$x}{pat} eq '1' and $S->{$y}{pat} eq '1';

  my $xstr = sprintf "$S->{$x}{mat}{name}\[%2d*$S->{$x}{mat}{name}N+$S->{$x}{mat}{name}n]", $S->{$x}{idx};
  my $ystr = sprintf "$S->{$y}{mat}{name}\[%2d*$S->{$y}{mat}{name}N+$S->{$y}{mat}{name}n]", $S->{$y}{idx};

  return $xstr if $S->{$y}{pat} eq '1';
  return $ystr if $S->{$x}{pat} eq '1';

  return "${xstr}*${ystr}";
}

sub multiply_gpu
{
  # Standard mutiplication - outputs unrolled C code, one line
  # per target matrix element.
  # Arguments: a, b, c   -- all GenMul::MBase with right dimensions.
  # Does:      c = a * b

  check_multiply_arguments(@_);

  my ($S, $a, $b, $c) = @_;

  my $is_c_symmetric = $c->isa("GenMul::MatrixSym");

  # With no_size_check matrices do not have to be compatible.
  my $k_max = $a->{N} <= $b->{M} ? $a->{N} : $b->{M};

  for (my $i = 0; $i < $c->{M}; ++$i)
  {
    my $j_max = $is_c_symmetric ?  $i + 1 : $c->{N};

    for (my $j = 0; $j < $j_max; ++$j)
    {
      my $x = $c->idx($i, $j);

      printf "$S->{prefix}$c->{name}\[%2d*$c->{name}N+$c->{name}n\] = ", $x;

      my @sum;

      for (my $k = 0; $k < $k_max; ++$k)
      {
        $S->generate_indices_and_patterns_for_multiplication($i, $j, $k);

        my $addend = $S->generate_addend_gpu('a', 'b');

        push @sum, $addend if defined $addend;
      }
      if (@sum)
      {
        print join(" + ", @sum), ";";
      }
      else
      {
        print "0;"
      }
      print "\n";
    }
  }

  $S->delete_temporaries();
}

# ----------------------------------------------------------------------

sub load_if_needed
{
  my ($S, $x) = @_;

  my $idx = $S->{$x}{idx};

  my $reg = $S->{$x}{mat}->reg_name($idx);

  if ($S->{$x}{cnt}[$idx] == 0)
  {
    $S->push_out("$S->{vectype} ${reg} = LD($S->{$x}{mat}{name}, $idx);");
    ++$S->{tick};
  }

  ++$S->{$x}{cnt}[$idx];

  return $reg;
}

sub store
{
  my ($S, $mat, $idx) = @_;

  my $reg = $mat->reg_name(${idx});

  $S->push_out("ST($mat->{name}, ${idx}, ${reg});");

  return $reg;
}

sub multiply_intrinsic
{
  check_multiply_arguments(@_);

  my ($S, $a, $b, $c) = @_;

  $S->{tick} = 0;

  $S->{out}  = [];

  # Counts of use. For a and b to fetch, for c to assign / add / mult / fma.
  # cc is used as tick at which store can be performed afterwards.
  my (@cc, @to_store);
  @cc = (0) x $c->mat_size();

  $S->{a}{cnt} = [ (0) x $a->mat_size() ];
  $S->{b}{cnt} = [ (0) x $b->mat_size() ];

  my $need_all_zeros = 0;
  my $need_all_ones  = 0;

  my $is_c_symmetric = $c->isa("GenMul::MatrixSym");

  # With no_size_check matrices do not have to be compatible.
  my $k_max = $a->{N} <= $b->{M} ? $a->{N} : $b->{M};

  for (my $i = 0; $i < $c->{M}; ++$i)
  {
    my $j_max = $is_c_symmetric ?  $i + 1 : $c->{N};

    for (my $k = 0; $k < $k_max; ++$k)
    {
      for (my $j = 0; $j < $j_max; ++$j)
      {
        my $x = $c->idx($i, $j);

        $S->generate_indices_and_patterns_for_multiplication($i, $j, $k);

        if ($S->{a}{pat} ne '0' and $S->{b}{pat} ne '0')
        {
          my ($areg, $breg, $sreg);

          if ($S->{a}{pat} eq '1' and $S->{b}{pat} eq '1')
          {
            $need_all_ones = 1;
            $sreg = "all_ones";
          }
          elsif ($S->{b}{pat} eq '1')
          {
            $sreg = $S->load_if_needed('a');
          }
          elsif ($S->{a}{pat} eq '1')
          {
            $sreg = $S->load_if_needed('b');
          }
          else
          {
            $areg = $S->load_if_needed('a');
            $breg = $S->load_if_needed('b');
          }

          my $creg = $c->reg_name($x);

          if ($cc[$x] == 0)
          {
            my $op = defined $sreg ? "${sreg}" : "MUL(${areg}, ${breg})";

            $S->push_out("$S->{vectype} ${creg} = ", $op, ";");
          }
          else
          {
            my $op = defined $sreg ?
                "ADD(${sreg}, ${creg})" :
                "FMA(${areg}, ${breg}, ${creg})";

            $S->push_out("${creg} = ", $op, ";");
          }

          ++$cc[$x];
          ++$S->{tick};
        }

        if ($k + 1 == $k_max)
        {
          if ($cc[$x] == 0)
          {
            $need_all_zeros = 1;

            $S->push_out("ST($c->{name}, $x, all_zeros);");
          }
          else
          {
            $cc[$x] = $S->{tick} + 4; #### Will be ready to store in 4 cycles. Really 4?
            push @to_store, $x;
          }
        }

        # Try to store the finished ones.
        while (1)
        {
          last unless @to_store;
          my $s = $to_store[0];
          last if $S->{tick} < $cc[$s];

          $S->store($c, $s);
          shift @to_store;
          ++$S->{tick};
        }

      }

      $S->push_out("") unless $i + 1 == $a->{M} and $k + 1 == $a->{N};
    }
  }

  for my $s (@to_store)
  {
    $S->store($c, $s);

    ++$S->{tick};
  }

  $S->handle_all_zeros_ones($need_all_zeros, $need_all_ones);

  for (@{$S->{out}})
  {
    print $S->{prefix} unless /^$/;
    print;
    print "\n";
  }

  $S->delete_temporaries();
}

# ----------------------------------------------------------------------

sub dump_multiply_std_and_intrinsic
{
  my ($S, $fname, $a, $b, $c) = @_;

  unless ($fname eq '-')
  {
    open FF, ">$fname";
    select FF;
  }

  print <<"FNORD";
#ifdef MPLEX_INTRINSICS

   for (int n = 0; n < N; n += MPLEX_INTRINSICS_WIDTH_BYTES / sizeof(T))
   {
FNORD

  $S->multiply_intrinsic($a, $b, $c);

  print <<"FNORD";
   }

#else

#pragma omp simd
   for (int n = 0; n < N; ++n)
   {
FNORD

  $S->multiply_standard($a, $b, $c);

  print <<"FNORD";
   }
#endif
FNORD

  unless ($fname eq '-')
  {
    close FF;
    select STDOUT;
  }
}

# ----------------------------------------------------------------------

sub dump_multiply_std_and_intrinsic_and_gpu
{
  my ($S, $fname, $a, $b, $c) = @_;

  unless ($fname eq '-')
  {
    open FF, ">$fname";
    select FF;
  }

  print <<"FNORD";
#ifndef __CUDACC__
#ifdef MPLEX_INTRINSICS

   for (int n = 0; n < N; n += MPLEX_INTRINSICS_WIDTH_BYTES / sizeof(T))
   {
FNORD

  $S->multiply_intrinsic($a, $b, $c);

  print <<"FNORD";
   }

#else

#pragma omp simd
   for (int n = 0; n < N; ++n)
   {
FNORD

  $S->multiply_standard($a, $b, $c);

  print <<"FNORD";
   }
#endif
#else  // __CUDACC__
FNORD
  $S->multiply_gpu($a, $b, $c);
  print <<"FNORD";
#endif  // __CUDACC__
FNORD

  unless ($fname eq '-')
  {
    close FF;
    select STDOUT;
  }
}

########################################################################
########################################################################
# THE END
########################################################################
########################################################################

1;
