#!/usr/bin/perl

@Offs[3] = [ 0, 1, 3, 1, 2, 4, 3, 4, 5 ];
@Offs[4] = [ 0, 1, 3, 6, 1, 2, 4, 7, 3, 4, 5, 8, 6, 7, 8, 9 ];
@Offs[5] = [ 0, 1, 3, 6, 10, 1, 2, 4, 7, 11, 3, 4, 5, 8, 12, 6, 7, 8, 9, 13, 10, 11, 12, 13, 14 ];
@Offs[6] = [ 0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20 ];

$PREF = "      ";
$BR   = " ";
$JOIN = "$BR";
$POST = " ";


#$a = "A.fArray";
#$b = "B.fArray";
#$c = "C.fArray";
$a = "a";
$b = "b";
$c = "c";

$vectype = "IntrVec_t";

$SYMMETRIC = 1;

################################################################################

sub mult_sym
{
  my $D = shift;

  my @Off = @{$Offs[$D]};

  for (my $i = 0; $i < $D; ++$i)
  {
    for (my $j = 0; $j < $D; ++$j)
    # for (my $j = 0; $j <= $i; ++$j)
    {
      # my $x = $Off[$i * $D + $j];
      my $x = $i * $D + $j;
      printf "${PREF}${c}[%2d*N+n] =${POST}", $x;

      my @sum;

      for (my $k = 0; $k < $D; ++$k)
      {
        my $iko = $Off[$i * $D + $k];
        my $kjo = $Off[$k * $D + $j];

        push @sum, sprintf("${a}[%2d*N+n]*${b}[%2d*N+n]", $iko, $kjo);
      }
      print join(" +$JOIN", @sum), ";";
      print "\n";
    }
  }
}

sub mult_sym_fma
{
  # This actually runs quite horribly, twice slower than the
  # fully expressd version.
  # Changing order of k and i loops doesn't change anything.
  #
  # However, this should be close to what we need for auto-generated
  # intrinsics!

  my $D = shift;

  my @Off = @{$Offs[$D]};

  for (my $i = 0; $i < $D; ++$i)
  {
    for (my $k = 0; $k < $D; ++$k)
    {
      for (my $j = 0; $j < $D; ++$j)
      {
        my $x = $i * $D + $j;
        my $iko = $Off[$i * $D + $k];
        my $kjo = $Off[$k * $D + $j];

        my $op = ($k == 0) ? " =" : "+=";

        printf "${PREF}${c}[%2d*N+n] ${op} ${a}[%2d*N+n]*${b}[%2d*N+n];\n",
           $x, $iko, $kjo;
      }
      print "\n";
    }
  }
}

sub reg_name
{
  my ($var, $idx) = @_;

  return "${var}_${idx}";

}

sub load_if_needed
{
  my ($var, $idx, $arc) = @_;

  my $reg = reg_name(${var}, ${idx});

  if ($arc->[$idx] == 0)
  {
    print "${PREF}${vectype} ${reg} = LD($var, $idx);\n";
    ++$tick;
  }

  ++$arc->[$idx];

  return $reg;
}

sub store
{
  my ($var, $idx) = @_;

  my $reg = reg_name(${var}, ${idx});

  print "${PREF}ST(${var}, ${idx}, ${reg});\n";

  return $reg;
}

sub mult_sym_fma_intrinsic
{
  # Order of k and i loops should be different for 3x3 and 6x6. (?)

  my $D = shift;

  my @Off = @{$Offs[$D]};

  local $tick = 0;

  # Counts of use. For a and b to fetch, for c to store
  my @ac, @bc, @cc, @to_store;

  for (my $i = 0; $i < $D; ++$i)
  {
    for (my $k = 0; $k < $D; ++$k)
    {
      for (my $j = 0; $j < $D; ++$j)
      {
        my $x = $i * $D + $j;
        my $iko = $Off[$i * $D + $k];
        my $kjo = $Off[$k * $D + $j];

        my $areg = load_if_needed("a", $iko, \@ac);
        my $breg = load_if_needed("b", $kjo, \@bc);
        my $creg = reg_name("c", $x);

        my $op = ($k == 0) ? "=" : "+=";

        if ($k == 0)
        {
          print "${PREF}${vectype} ${creg} = MUL(${areg}, ${breg});\n";
        }
        else
        {
          print "${PREF}${creg} = FMA(${areg}, ${breg}, ${creg});\n";
        }

        ++$tick;

        if ($k + 1 == $D)
        {
          $cc[$x] = $tick + 4; #### Will be ready to store in 4 cycles. Really 4?
          push @to_store, $x;
        }

        # Try to store the finished ones.
        while (1)
        {
          last unless @to_store;
          my $s = $to_store[0];
          last if $tick < $cc[$s];

          store("c", $s);
          shift @to_store;
          ++$tick;
        }

      }
      print "\n";
    }
  }

  for $s (@to_store)
  {
    store("c", $s);

    ++$tick;
  }
}

################################################################################

sub mult_std
{
  my $D = shift;

  for (my $i = 0; $i < $D; ++$i)
  {
    for (my $j = 0; $j < $D; ++$j)
    {
      my $x = $i * $D + $j;
      printf "${PREF}${c}[%2d*N+n] =${POST}", $x;

      my @sum;

      for (my $k = 0; $k < $D; ++$k)
      {
        my $iko = $i * $D + $k;
        my $kjo = $k * $D + $j;

        push @sum, sprintf "${a}[%2d*N+n]*${b}[%2d*N+n]", $iko, $kjo;
      }
      print join(" +$JOIN", @sum), ";";
      print "\n";
    }
  }
}

################################################################################

if (scalar @ARGV != 1)
{
  print STDERR "Usage: $0 function_call\n";
  print STDERR << "FNORD";
Some options:
  $0 "mult_sym(3);"
  $0 "mult_sym_fma(3);"
  $0 "mult_sym_fma_intrinsic(6);"

  $0 "mult_std();"
FNORD

  exit(1);
}

eval $ARGV[0];
