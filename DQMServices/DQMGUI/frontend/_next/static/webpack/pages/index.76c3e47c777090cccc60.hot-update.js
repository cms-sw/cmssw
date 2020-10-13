webpackHotUpdate_N_E("pages/index",{

/***/ "./components/Nav.tsx":
/*!****************************!*\
  !*** ./components/Nav.tsx ***!
  \****************************/
/*! exports provided: Nav, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Nav", function() { return Nav; });
/* harmony import */ var _babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/extends */ "./node_modules/@babel/runtime/helpers/esm/extends.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _searchButton__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./searchButton */ "./components/searchButton.tsx");
/* harmony import */ var _helpButton__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./helpButton */ "./components/helpButton.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../config/config */ "./config/config.ts");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/Nav.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;






var Nav = function Nav(_ref) {
  _s();

  var initial_search_run_number = _ref.initial_search_run_number,
      initial_search_dataset_name = _ref.initial_search_dataset_name,
      setRunNumber = _ref.setRunNumber,
      setDatasetName = _ref.setDatasetName,
      handler = _ref.handler,
      type = _ref.type,
      defaultRunNumber = _ref.defaultRunNumber,
      defaultDatasetName = _ref.defaultDatasetName;

  var _Form$useForm = antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm(),
      _Form$useForm2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_Form$useForm, 1),
      form = _Form$useForm2[0];

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_run_number || ''),
      form_search_run_number = _useState[0],
      setFormRunNumber = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_dataset_name || ''),
      form_search_dataset_name = _useState2[0],
      setFormDatasetName = _useState2[1]; // We have to wait for changin initial_search_run_number and initial_search_dataset_name coming from query, because the first render they are undefined and therefore the initialValues doesn't grab them


  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    form.resetFields();
    setFormRunNumber(initial_search_run_number || '');
    setFormDatasetName(initial_search_dataset_name || '');
  }, [initial_search_run_number, initial_search_dataset_name, form]);
  var layout = {
    labelCol: {
      span: 8
    },
    wrapperCol: {
      span: 16
    }
  };
  var tailLayout = {
    wrapperCol: {
      offset: 0,
      span: 4
    }
  };
  return __jsx("div", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 54,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomForm"], Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({
    form: form,
    layout: 'inline',
    justifycontent: "center",
    width: "max-content"
  }, layout, {
    name: "search_form".concat(type),
    className: "fieldLabel",
    initialValues: {
      run_number: initial_search_run_number,
      dataset_name: initial_search_dataset_name
    },
    onFinish: function onFinish() {
      setRunNumber && setRunNumber(form_search_run_number);
      setDatasetName && setDatasetName(form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 55,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 72,
      columnNumber: 9
    }
  }), __jsx(_helpButton__WEBPACK_IMPORTED_MODULE_6__["QuestionButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "run_number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 75,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "run_number",
    onChange: function onChange(e) {
      return setFormRunNumber(e.target.value);
    },
    placeholder: "Enter run number",
    type: "text",
    name: "run_number",
    value: defaultRunNumber,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 76,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "dataset_name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 87,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "dataset_name",
    placeholder: "Enter dataset name",
    onChange: function onChange(e) {
      return setFormDatasetName(e.target.value);
    },
    type: "text",
    value: defaultDatasetName,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 88,
      columnNumber: 11
    }
  })), _config_config__WEBPACK_IMPORTED_MODULE_7__["functions_config"].new_back_end.lumisections_on && __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "lumisection",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 100,
      columnNumber: 11
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "lumisection",
    placeholder: "Enter lumisection",
    onChange: function onChange(e) {
      return setFormDatasetName(e.target.value);
    },
    type: "text",
    value: defaultDatasetName,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 101,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 112,
      columnNumber: 9
    }
  }), __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_5__["SearchButton"], {
    onClick: function onClick() {
      return handler(form_search_run_number, form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 113,
      columnNumber: 11
    }
  }))));
};

_s(Nav, "d/o1hn25bH6EF0LAvbTEx8d/DOY=", false, function () {
  return [antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm];
});

_c = Nav;
/* harmony default export */ __webpack_exports__["default"] = (Nav);

var _c;

$RefreshReg$(_c, "Nav");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9OYXYudHN4Il0sIm5hbWVzIjpbIk5hdiIsImluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIiLCJpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRSdW5OdW1iZXIiLCJzZXREYXRhc2V0TmFtZSIsImhhbmRsZXIiLCJ0eXBlIiwiZGVmYXVsdFJ1bk51bWJlciIsImRlZmF1bHREYXRhc2V0TmFtZSIsIkZvcm0iLCJ1c2VGb3JtIiwiZm9ybSIsInVzZVN0YXRlIiwiZm9ybV9zZWFyY2hfcnVuX251bWJlciIsInNldEZvcm1SdW5OdW1iZXIiLCJmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRGb3JtRGF0YXNldE5hbWUiLCJ1c2VFZmZlY3QiLCJyZXNldEZpZWxkcyIsImxheW91dCIsImxhYmVsQ29sIiwic3BhbiIsIndyYXBwZXJDb2wiLCJ0YWlsTGF5b3V0Iiwib2Zmc2V0IiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsImUiLCJ0YXJnZXQiLCJ2YWx1ZSIsImZ1bmN0aW9uc19jb25maWciLCJuZXdfYmFja19lbmQiLCJsdW1pc2VjdGlvbnNfb24iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBYU8sSUFBTUEsR0FBRyxHQUFHLFNBQU5BLEdBQU0sT0FTSDtBQUFBOztBQUFBLE1BUmRDLHlCQVFjLFFBUmRBLHlCQVFjO0FBQUEsTUFQZEMsMkJBT2MsUUFQZEEsMkJBT2M7QUFBQSxNQU5kQyxZQU1jLFFBTmRBLFlBTWM7QUFBQSxNQUxkQyxjQUtjLFFBTGRBLGNBS2M7QUFBQSxNQUpkQyxPQUljLFFBSmRBLE9BSWM7QUFBQSxNQUhkQyxJQUdjLFFBSGRBLElBR2M7QUFBQSxNQUZkQyxnQkFFYyxRQUZkQSxnQkFFYztBQUFBLE1BRGRDLGtCQUNjLFFBRGRBLGtCQUNjOztBQUFBLHNCQUNDQyx5Q0FBSSxD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O0FBQUE7QUFBQSxJQURGLENBaENGLEVBNENJc0IsK0RBQWdCLENBQUNDLFlBQWpCLENBQThCQyxlQUE5QixJQUNBLE1BQUMsZ0VBQUQ7QUFBZ0IsUUFBSSxFQUFDLGFBQXJCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDZEQUFEO0FBQ0UsTUFBRSxFQUFDLGFBREw7QUFFRSxlQUFXLEVBQUMsbUJBRmQ7QUFHRSxZQUFRLEVBQUUsa0JBQUNMLENBQUQ7QUFBQSxhQUNSWCxrQkFBa0IsQ0FBQ1csQ0FBQyxDQUFDQyxNQUFGLENBQVNDLEtBQVYsQ0FEVjtBQUFBLEtBSFo7QUFNRSxRQUFJLEVBQUMsTUFOUDtBQU9FLFNBQUssRUFBRXJCLGtCQVBUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQTdDSixFQXlERSxNQUFDLHlDQUFELENBQU0sSUFBTix5RkFBZWUsVUFBZjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BQ0UsTUFBQywwREFBRDtBQUNFLFdBQU8sRUFBRTtBQUFBLGFBQ1BsQixPQUFPLENBQUNRLHNCQUFELEVBQXlCRSx3QkFBekIsQ0FEQTtBQUFBLEtBRFg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBekRGLENBREYsQ0FERjtBQXFFRCxDQXRHTTs7R0FBTWYsRztVQVVJUyx5Q0FBSSxDQUFDQyxPOzs7S0FWVFYsRztBQXdHRUEsa0VBQWYiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNzZjM2U0N2M3NzcwOTBjY2NjNjAuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyBDaGFuZ2VFdmVudCwgRGlzcGF0Y2gsIHVzZUVmZmVjdCwgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XG5cbmltcG9ydCB7IFN0eWxlZEZvcm1JdGVtLCBTdHlsZWRJbnB1dCwgQ3VzdG9tRm9ybSB9IGZyb20gJy4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBTZWFyY2hCdXR0b24gfSBmcm9tICcuL3NlYXJjaEJ1dHRvbic7XG5pbXBvcnQgeyBRdWVzdGlvbkJ1dHRvbiB9IGZyb20gJy4vaGVscEJ1dHRvbic7XG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vY29uZmlnL2NvbmZpZyc7XG5cbmludGVyZmFjZSBOYXZQcm9wcyB7XG4gIHNldFJ1bk51bWJlcj86IERpc3BhdGNoPGFueT47XG4gIHNldERhdGFzZXROYW1lPzogRGlzcGF0Y2g8YW55PjtcbiAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcj86IHN0cmluZztcbiAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPzogc3RyaW5nO1xuICBoYW5kbGVyKHNlYXJjaF9ieV9ydW5fbnVtYmVyOiBzdHJpbmcsIHNlYXJjaF9ieV9kYXRhc2V0X25hbWU6IHN0cmluZyk6IHZvaWQ7XG4gIHR5cGU6IHN0cmluZztcbiAgZGVmYXVsdFJ1bk51bWJlcj86IHVuZGVmaW5lZCB8IHN0cmluZztcbiAgZGVmYXVsdERhdGFzZXROYW1lPzogc3RyaW5nIHwgdW5kZWZpbmVkO1xufVxuXG5leHBvcnQgY29uc3QgTmF2ID0gKHtcbiAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcixcbiAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lLFxuICBzZXRSdW5OdW1iZXIsXG4gIHNldERhdGFzZXROYW1lLFxuICBoYW5kbGVyLFxuICB0eXBlLFxuICBkZWZhdWx0UnVuTnVtYmVyLFxuICBkZWZhdWx0RGF0YXNldE5hbWUsXG59OiBOYXZQcm9wcykgPT4ge1xuICBjb25zdCBbZm9ybV0gPSBGb3JtLnVzZUZvcm0oKTtcbiAgY29uc3QgW2Zvcm1fc2VhcmNoX3J1bl9udW1iZXIsIHNldEZvcm1SdW5OdW1iZXJdID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciB8fCAnJ1xuICApO1xuICBjb25zdCBbZm9ybV9zZWFyY2hfZGF0YXNldF9uYW1lLCBzZXRGb3JtRGF0YXNldE5hbWVdID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lIHx8ICcnXG4gICk7XG5cbiAgLy8gV2UgaGF2ZSB0byB3YWl0IGZvciBjaGFuZ2luIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgYW5kIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSBjb21pbmcgZnJvbSBxdWVyeSwgYmVjYXVzZSB0aGUgZmlyc3QgcmVuZGVyIHRoZXkgYXJlIHVuZGVmaW5lZCBhbmQgdGhlcmVmb3JlIHRoZSBpbml0aWFsVmFsdWVzIGRvZXNuJ3QgZ3JhYiB0aGVtXG4gIHVzZUVmZmVjdCgoKSA9PiB7XG4gICAgZm9ybS5yZXNldEZpZWxkcygpO1xuICAgIHNldEZvcm1SdW5OdW1iZXIoaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciB8fCAnJyk7XG4gICAgc2V0Rm9ybURhdGFzZXROYW1lKGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSB8fCAnJyk7XG4gIH0sIFtpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyLCBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUsIGZvcm1dKTtcblxuICBjb25zdCBsYXlvdXQgPSB7XG4gICAgbGFiZWxDb2w6IHsgc3BhbjogOCB9LFxuICAgIHdyYXBwZXJDb2w6IHsgc3BhbjogMTYgfSxcbiAgfTtcbiAgY29uc3QgdGFpbExheW91dCA9IHtcbiAgICB3cmFwcGVyQ29sOiB7IG9mZnNldDogMCwgc3BhbjogNCB9LFxuICB9O1xuXG4gIHJldHVybiAoXG4gICAgPGRpdj5cbiAgICAgIDxDdXN0b21Gb3JtXG4gICAgICAgIGZvcm09e2Zvcm19XG4gICAgICAgIGxheW91dD17J2lubGluZSd9XG4gICAgICAgIGp1c3RpZnljb250ZW50PVwiY2VudGVyXCJcbiAgICAgICAgd2lkdGg9XCJtYXgtY29udGVudFwiXG4gICAgICAgIHsuLi5sYXlvdXR9XG4gICAgICAgIG5hbWU9e2BzZWFyY2hfZm9ybSR7dHlwZX1gfVxuICAgICAgICBjbGFzc05hbWU9XCJmaWVsZExhYmVsXCJcbiAgICAgICAgaW5pdGlhbFZhbHVlcz17e1xuICAgICAgICAgIHJ1bl9udW1iZXI6IGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIsXG4gICAgICAgICAgZGF0YXNldF9uYW1lOiBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUsXG4gICAgICAgIH19XG4gICAgICAgIG9uRmluaXNoPXsoKSA9PiB7XG4gICAgICAgICAgc2V0UnVuTnVtYmVyICYmIHNldFJ1bk51bWJlcihmb3JtX3NlYXJjaF9ydW5fbnVtYmVyKTtcbiAgICAgICAgICBzZXREYXRhc2V0TmFtZSAmJiBzZXREYXRhc2V0TmFtZShmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUpO1xuICAgICAgICB9fVxuICAgICAgPlxuICAgICAgICA8Rm9ybS5JdGVtIHsuLi50YWlsTGF5b3V0fT5cbiAgICAgICAgICA8UXVlc3Rpb25CdXR0b24gLz5cbiAgICAgICAgPC9Gb3JtLkl0ZW0+XG4gICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPVwicnVuX251bWJlclwiPlxuICAgICAgICAgIDxTdHlsZWRJbnB1dFxuICAgICAgICAgICAgaWQ9XCJydW5fbnVtYmVyXCJcbiAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogQ2hhbmdlRXZlbnQ8SFRNTElucHV0RWxlbWVudD4pID0+XG4gICAgICAgICAgICAgIHNldEZvcm1SdW5OdW1iZXIoZS50YXJnZXQudmFsdWUpXG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBwbGFjZWhvbGRlcj1cIkVudGVyIHJ1biBudW1iZXJcIlxuICAgICAgICAgICAgdHlwZT1cInRleHRcIlxuICAgICAgICAgICAgbmFtZT1cInJ1bl9udW1iZXJcIlxuICAgICAgICAgICAgdmFsdWU9e2RlZmF1bHRSdW5OdW1iZXJ9XG4gICAgICAgICAgLz5cbiAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtIG5hbWU9XCJkYXRhc2V0X25hbWVcIj5cbiAgICAgICAgICA8U3R5bGVkSW5wdXRcbiAgICAgICAgICAgIGlkPVwiZGF0YXNldF9uYW1lXCJcbiAgICAgICAgICAgIHBsYWNlaG9sZGVyPVwiRW50ZXIgZGF0YXNldCBuYW1lXCJcbiAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogQ2hhbmdlRXZlbnQ8SFRNTElucHV0RWxlbWVudD4pID0+XG4gICAgICAgICAgICAgIHNldEZvcm1EYXRhc2V0TmFtZShlLnRhcmdldC52YWx1ZSlcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHR5cGU9XCJ0ZXh0XCJcbiAgICAgICAgICAgIHZhbHVlPXtkZWZhdWx0RGF0YXNldE5hbWV9XG4gICAgICAgICAgLz5cbiAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cbiAgICAgICAge1xuICAgICAgICAgIGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLmx1bWlzZWN0aW9uc19vbiAmJlxuICAgICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPVwibHVtaXNlY3Rpb25cIj5cbiAgICAgICAgICAgIDxTdHlsZWRJbnB1dFxuICAgICAgICAgICAgICBpZD1cImx1bWlzZWN0aW9uXCJcbiAgICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBsdW1pc2VjdGlvblwiXG4gICAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogQ2hhbmdlRXZlbnQ8SFRNTElucHV0RWxlbWVudD4pID0+XG4gICAgICAgICAgICAgICAgc2V0Rm9ybURhdGFzZXROYW1lKGUudGFyZ2V0LnZhbHVlKVxuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIHR5cGU9XCJ0ZXh0XCJcbiAgICAgICAgICAgICAgdmFsdWU9e2RlZmF1bHREYXRhc2V0TmFtZX1cbiAgICAgICAgICAgIC8+XG4gICAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cbiAgICAgICAgfVxuICAgICAgICA8Rm9ybS5JdGVtIHsuLi50YWlsTGF5b3V0fT5cbiAgICAgICAgICA8U2VhcmNoQnV0dG9uXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PlxuICAgICAgICAgICAgICBoYW5kbGVyKGZvcm1fc2VhcmNoX3J1bl9udW1iZXIsIGZvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSlcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAvPlxuICAgICAgICA8L0Zvcm0uSXRlbT5cbiAgICAgIDwvQ3VzdG9tRm9ybT5cbiAgICA8L2Rpdj5cbiAgKTtcbn07XG5cbmV4cG9ydCBkZWZhdWx0IE5hdjtcbiJdLCJzb3VyY2VSb290IjoiIn0=