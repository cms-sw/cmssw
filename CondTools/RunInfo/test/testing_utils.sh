#!/bin/bash

# Prints a failure message, optionally dumps a log file, and exits with the given status code.
function die {
    message="$1"
    status="$2"
    log_file="$3"

    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "Log output:"
        cat "$log_file"
    fi

    echo "Failure $message: status $status"
    exit "$status"
}

# Compares two values and fails the test if they differ.
function assert_equal {
    expected="$1"
    actual="$2"
    message="$3"
    log_file="$4"

    if [ "$expected" != "$actual" ]; then
        die "$message (expected $expected, got $actual)" 1 "$log_file"
    fi
}

