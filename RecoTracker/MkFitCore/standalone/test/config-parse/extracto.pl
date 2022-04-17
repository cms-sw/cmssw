#!/usr/bin/perl -n

if (m/^(class|struct)\s(\w+)/)
{
    my $soc = $1;
    my $cls = $2;
    push @c, $cls if ($soc eq class and $cls =~ m/Iteration/);
}

END
{
    print "// For ConfigLinkDef.h\n";
    print map { "#pragma link C++ class mkfit::$_;\n" } @c;

    print "\n// For dictgen:\n";
    print "std::vector<std::string> classes = {\n";
    print join(",\n", map { "  \"mkfit::$_\"" } @c);
    print "\n};\n";
}
