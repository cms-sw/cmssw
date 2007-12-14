package CMSDetID;

use strict;
use warnings;

require Exporter;

our @ISA = qw(Exporter);

# Items to export into callers namespace by default. Note: do not export
# names by default without a very good reason. Use EXPORT_OK instead.
# Do not simply export all your public functions/methods/constants.

# This allows declaration       use Foo::Bar ':all';
# If you do not need this, moving things directly into @EXPORT or @EXPORT_OK
# will save memory.
our %EXPORT_TAGS = ( 'all' => [ qw(
) ] );

our @EXPORT_OK = ( @{ $EXPORT_TAGS{'all'} } );

our @EXPORT = qw(

);

our $VERSION = '0.01';


# Preloaded methods go here.

sub new {
      my $class = shift;
      my $detid = shift;
      bless \$detid, $class;
}
sub det {
    my $self = shift; my $detid = $$self;
    #print "MyDetID is $detid\n";
    return ($detid >> 28) & (0xF);
}
sub subdet {
    my $self = shift; my $detid = $$self;
    return ($detid >> 25) & (0x7);
}
my %subdetNames = (3 => 'TIB', 4=>'TID', 5=>'TOB', 6=>'TEC' );
sub subdetName { return $subdetNames{shift->subdet()}; }

sub tiblayer {
    my $self = shift; my $detid = $$self;
    return ($detid >> 14) & 0x7;
}

# Autoload methods go after =cut, and are processed by the autosplit program.

1;
__END__
